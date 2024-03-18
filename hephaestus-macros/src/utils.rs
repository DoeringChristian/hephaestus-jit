use proc_macro2::{Span, TokenStream, TokenTree};
use quote::quote;
use syn::Ident;

pub fn crate_name() -> proc_macro2::TokenStream {
    let found_crate = proc_macro_crate::crate_name("hephaestus-jit").unwrap();
    match found_crate {
        proc_macro_crate::FoundCrate::Itself => quote!(crate),
        proc_macro_crate::FoundCrate::Name(name) => {
            let ident = Ident::new(&name, Span::call_site());
            quote!(#ident)
        }
    }
}
