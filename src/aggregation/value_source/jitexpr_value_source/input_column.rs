//! The fast-field columns bound to the inputs of a compiled jitexpr expression.

use std::sync::Arc;

use columnar::{
    Column, Dictionary, DynamicColumn, DynamicColumnHandle, MonotonicallyMappableToU64,
};
use jitexpr::types::{VarType, VariableValue};

use crate::query::doc_predicate_query::var_type_for_column_type;
use crate::{DocId, TantivyError, COLLECT_BLOCK_BUFFER_LEN};

/// A fast-field column bound to an input of the compiled expression.
pub(super) struct InputColumn {
    var_type: VarType,
    /// Values in their monotonic `u64` mapping. Term ords for a `Str` input.
    column: Column<u64>,
    /// The dictionary of the term ords. `Some` if and only if `var_type` is `Str`.
    term_dictionary: Option<Arc<Dictionary>>,
    /// `first_values[i]` is the first value of `docs[i]`, for the docs of the last block.
    first_values: Vec<Option<u64>>,
    /// The terms of `first_values`, for a `Str` input.
    block_terms: BlockTerms,
}

impl InputColumn {
    /// Opens the column of `handle`. The type of the input is given by the column type.
    pub(super) fn open(handle: &DynamicColumnHandle) -> crate::Result<InputColumn> {
        let column_type = handle.column_type();
        let Some(var_type) = var_type_for_column_type(column_type) else {
            return Err(TantivyError::InternalError(format!(
                "a column of type {column_type} cannot be bound to an input"
            )));
        };
        if var_type == VarType::Str {
            let DynamicColumn::Str(str_column) = handle.open()? else {
                return Err(TantivyError::InternalError(
                    "a text column could not be opened as a text column".to_string(),
                ));
            };
            let (term_dictionary, term_ords) = str_column.into_parts();
            Ok(InputColumn {
                var_type,
                column: term_ords,
                term_dictionary: Some(term_dictionary),
                first_values: Vec::with_capacity(COLLECT_BLOCK_BUFFER_LEN),
                block_terms: BlockTerms::default(),
            })
        } else {
            let Some(column) = handle.open_u64_lenient()? else {
                return Err(TantivyError::InternalError(format!(
                    "a column of type {column_type} could not be opened as u64"
                )));
            };
            Ok(InputColumn {
                var_type,
                column,
                term_dictionary: None,
                first_values: Vec::with_capacity(COLLECT_BLOCK_BUFFER_LEN),
                block_terms: BlockTerms::default(),
            })
        }
    }

    pub(super) fn var_type(&self) -> VarType {
        self.var_type
    }

    /// Loads the first value of each doc of `docs`.
    ///
    /// Panics if the dictionary of a text column is corrupted.
    pub(super) fn load_block(&mut self, docs: &[DocId]) {
        // `first_vals` does not write the entries of the docs without a value: we reset them.
        self.first_values.clear();
        self.first_values.resize(docs.len(), None);
        self.column.first_vals(docs, &mut self.first_values);
        if let Some(term_dictionary) = &self.term_dictionary {
            self.block_terms.load(term_dictionary, &self.first_values);
        }
    }

    /// Returns the value of `docs[doc_pos]`, for the `docs` of the last `load_block` call.
    #[inline]
    pub(super) fn value(&self, doc_pos: usize) -> VariableValue<'_> {
        let Some(value) = self.first_values[doc_pos] else {
            return VariableValue::none();
        };
        match self.var_type {
            VarType::Bool => VariableValue::from(bool::from_u64(value)),
            VarType::I64 => VariableValue::from(i64::from_u64(value)),
            VarType::U64 => VariableValue::from(value),
            VarType::F64 => VariableValue::from(f64::from_u64(value)),
            VarType::Str => VariableValue::from(self.block_terms.term(value)),
            VarType::None => VariableValue::none(),
        }
    }
}

/// The terms of the term ords of a block, fetched with a single dictionary lookup.
#[derive(Default)]
struct BlockTerms {
    /// The distinct term ords of the block, sorted.
    sorted_ords: Vec<u64>,
    /// The concatenated terms, in the order of `sorted_ords`.
    buffer: String,
    /// `buffer[offsets[i]..offsets[i + 1]]` is the term of `sorted_ords[i]`.
    offsets: Vec<usize>,
}

impl BlockTerms {
    /// Panics if the dictionary is corrupted: `load_block` cannot return an error.
    fn load(&mut self, term_dictionary: &Dictionary, term_ords: &[Option<u64>]) {
        let BlockTerms {
            sorted_ords,
            buffer,
            offsets,
        } = self;
        sorted_ords.clear();
        sorted_ords.extend(term_ords.iter().flatten());
        sorted_ords.sort_unstable();
        sorted_ords.dedup();
        buffer.clear();
        offsets.clear();
        offsets.push(0);
        let all_found = term_dictionary
            .sorted_ords_to_term_cb(sorted_ords, |term| {
                let term_str =
                    std::str::from_utf8(term).expect("fast-field string dictionary is corrupted");
                buffer.push_str(term_str);
                offsets.push(buffer.len());
            })
            .expect("fast-field string dictionary is corrupted");
        assert!(all_found, "fast-field string dictionary is corrupted");
    }

    /// Precondition: `term_ord` is one of the term ords of the last `load` call.
    #[inline]
    fn term(&self, term_ord: u64) -> &str {
        let idx = self
            .sorted_ords
            .binary_search(&term_ord)
            .expect("the term ord belongs to the block");
        &self.buffer[self.offsets[idx]..self.offsets[idx + 1]]
    }
}
