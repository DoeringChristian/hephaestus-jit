use syn::{parse_macro_input, DeriveInput};

use self::traits::{derive_as_var_type_impl, derive_construct_impl, derive_traverse_impl};

mod traits;

#[proc_macro_derive(AsVarType)]
pub fn deive_as_var_type(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    derive_as_var_type_impl(parse_macro_input!(input as DeriveInput)).into()
}

#[proc_macro_derive(Traverse)]
pub fn derive_traverse(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    derive_traverse_impl(parse_macro_input!(input as DeriveInput)).into()
}

#[proc_macro_derive(Construct)]
pub fn derive_construct(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    derive_construct_impl(parse_macro_input!(input as DeriveInput)).into()
}
