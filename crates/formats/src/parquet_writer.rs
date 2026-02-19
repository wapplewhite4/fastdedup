//! Streaming Parquet writer
//!
//! Provides batch writing to Parquet files with automatic schema inference
//! from JSON values.  The parquet footer (required for a valid file) is
//! written when `close()` is called — if this is never called the output
//! file will be corrupt.

use crate::{Error, Record, Result};
use arrow::array::*;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use serde_json::{Map, Value};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

const DEFAULT_BATCH_SIZE: usize = 4096;

/// Streaming Parquet writer that accumulates records and writes in batches.
///
/// Schema is inferred from the first batch of records.  Call `close()` when
/// finished to flush remaining records and write the parquet footer.
pub struct ParquetWriter {
    /// Lazily initialised once the schema is known (after first flush).
    arrow_writer: Option<ArrowWriter<File>>,
    /// Inferred schema, set on first flush.
    schema: Option<Arc<Schema>>,
    /// Records buffered until the next batch flush.
    pending: Vec<Map<String, Value>>,
    batch_size: usize,
    /// Held until the first flush so we can pass it to ArrowWriter::try_new.
    file: Option<File>,
}

impl ParquetWriter {
    /// Open a new Parquet file for writing at `path`.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::create(path)?;
        Ok(Self {
            arrow_writer: None,
            schema: None,
            pending: Vec::with_capacity(DEFAULT_BATCH_SIZE),
            batch_size: DEFAULT_BATCH_SIZE,
            file: Some(file),
        })
    }

    /// Buffer a record for writing.  Automatically flushes a batch when the
    /// internal buffer reaches `batch_size`.
    pub fn write_record(&mut self, record: &Record) -> Result<()> {
        if let Value::Object(map) = &record.data {
            self.pending.push(map.clone());
            if self.pending.len() >= self.batch_size {
                self.flush_pending()?;
            }
        }
        Ok(())
    }

    /// Flush remaining buffered records and write the parquet footer.
    ///
    /// This **must** be called to produce a valid parquet file.
    pub fn close(mut self) -> Result<()> {
        self.flush_pending()?;
        if let Some(writer) = self.arrow_writer {
            writer.close()?;
        } else if let Some(file) = self.file {
            // No records were ever written — emit a valid empty parquet file.
            let schema = Arc::new(Schema::empty());
            let props = WriterProperties::builder().build();
            let writer = ArrowWriter::try_new(file, schema, Some(props))?;
            writer.close()?;
        }
        Ok(())
    }

    fn flush_pending(&mut self) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }

        // Lazily initialise the ArrowWriter on first flush once schema can be inferred.
        if self.schema.is_none() {
            let schema = infer_schema(&self.pending);
            let file = self
                .file
                .take()
                .expect("file must be present before first flush");
            let props = WriterProperties::builder().build();
            let writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;
            self.arrow_writer = Some(writer);
            self.schema = Some(schema);
        }

        let schema = self.schema.as_ref().unwrap().clone();
        let batch = records_to_batch(&self.pending, &schema)?;
        if let Some(w) = self.arrow_writer.as_mut() {
            w.write(&batch)?;
        }
        self.pending.clear();
        Ok(())
    }
}

/// Infer an Arrow schema from a non-empty slice of JSON records.
///
/// Field order follows insertion order of the first record, with any
/// additional fields from subsequent records appended.
fn infer_schema(records: &[Map<String, Value>]) -> Arc<Schema> {
    let mut field_names: Vec<String> = Vec::new();
    for record in records {
        for key in record.keys() {
            if !field_names.contains(key) {
                field_names.push(key.clone());
            }
        }
    }

    let fields: Vec<Field> = field_names
        .iter()
        .map(|name| {
            let dtype = infer_field_type(records, name);
            Field::new(name, dtype, true /* nullable */)
        })
        .collect();

    Arc::new(Schema::new(fields))
}

/// Choose the Arrow DataType for a JSON field by scanning a slice of records.
///
/// Rules (in priority order):
/// - Any string value → `Utf8`
/// - Any complex value (array/object) → `Utf8` (stringified)
/// - Float-only numbers → `Float64`
/// - Int-only numbers → `Int64`
/// - Bool-only → `Boolean`
/// - Mixed or all-null → `Utf8`
fn infer_field_type(records: &[Map<String, Value>], field: &str) -> DataType {
    let mut has_float = false;
    let mut has_int = false;
    let mut has_bool = false;
    let mut has_string = false;

    for record in records {
        match record.get(field) {
            Some(Value::Bool(_)) => has_bool = true,
            Some(Value::Number(n)) => {
                // serde_json represents 1.0 as f64 but also matches is_i64/is_u64.
                // Treat as float only if it has a fractional component.
                if n.is_f64() && !n.is_i64() && !n.is_u64() {
                    has_float = true;
                } else {
                    has_int = true;
                }
            }
            Some(Value::String(_)) => has_string = true,
            Some(Value::Null) | None => {}
            _ => has_string = true, // arrays, objects → stringify
        }
    }

    if has_string {
        return DataType::Utf8;
    }
    if has_float && !has_int && !has_bool {
        return DataType::Float64;
    }
    if has_int && !has_float && !has_bool {
        return DataType::Int64;
    }
    if has_bool && !has_int && !has_float {
        return DataType::Boolean;
    }
    // Mixed numeric/bool types or all-null → stringify
    DataType::Utf8
}

/// Convert a slice of JSON records to an Arrow `RecordBatch` using `schema`.
fn records_to_batch(records: &[Map<String, Value>], schema: &Arc<Schema>) -> Result<RecordBatch> {
    let num_rows = records.len();
    let mut columns: Vec<Arc<dyn Array>> = Vec::with_capacity(schema.fields().len());

    for field in schema.fields() {
        let col = build_column(records, field.name(), field.data_type(), num_rows)?;
        columns.push(col);
    }

    RecordBatch::try_new(schema.clone(), columns).map_err(Error::ArrowError)
}

fn build_column(
    records: &[Map<String, Value>],
    field_name: &str,
    dtype: &DataType,
    num_rows: usize,
) -> Result<Arc<dyn Array>> {
    match dtype {
        DataType::Boolean => {
            let mut b = BooleanBuilder::with_capacity(num_rows);
            for rec in records {
                match rec.get(field_name) {
                    Some(Value::Bool(v)) => b.append_value(*v),
                    _ => b.append_null(),
                }
            }
            Ok(Arc::new(b.finish()))
        }
        DataType::Int64 => {
            let mut b = Int64Builder::with_capacity(num_rows);
            for rec in records {
                match rec.get(field_name) {
                    Some(Value::Number(n)) => match n.as_i64() {
                        Some(i) => b.append_value(i),
                        None => b.append_null(),
                    },
                    _ => b.append_null(),
                }
            }
            Ok(Arc::new(b.finish()))
        }
        DataType::Float64 => {
            let mut b = Float64Builder::with_capacity(num_rows);
            for rec in records {
                match rec.get(field_name) {
                    Some(Value::Number(n)) => match n.as_f64() {
                        Some(f) => b.append_value(f),
                        None => b.append_null(),
                    },
                    _ => b.append_null(),
                }
            }
            Ok(Arc::new(b.finish()))
        }
        _ => {
            // Utf8 and any unrecognised type: store as string.
            let mut b = StringBuilder::with_capacity(num_rows, num_rows * 64);
            for rec in records {
                match rec.get(field_name) {
                    Some(Value::String(s)) => b.append_value(s),
                    Some(Value::Null) | None => b.append_null(),
                    Some(other) => b.append_value(&other.to_string()),
                }
            }
            Ok(Arc::new(b.finish()))
        }
    }
}
