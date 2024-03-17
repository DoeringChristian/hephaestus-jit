use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::{DeriveInput, Ident};

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

pub fn derive_as_var_type_impl(input: DeriveInput) -> TokenStream {
    // determine if the struct is repr C
    let repr_c = input
        .attrs
        .iter()
        .find(|attr| {
            if attr.path().is_ident("repr") {
                let mut repr_c = false;
                attr.parse_nested_meta(|meta| {
                    if meta.path.is_ident("C") {
                        repr_c = true;
                        return Ok(());
                    }
                    Err(meta.error("unrecognized repr"))
                })
                .unwrap();
                return repr_c;
            }
            false
        })
        .is_some();
    assert!(repr_c, "AsVarType requires the struct to be repr C!");

    // Get the types of the struct
    let types = match &input.data {
        syn::Data::Struct(data) => match &data.fields {
            syn::Fields::Named(fields) => fields
                .named
                .iter()
                .map(|field| &field.ty)
                .collect::<Vec<_>>(),
            syn::Fields::Unnamed(fields) => fields
                .unnamed
                .iter()
                .map(|field| &field.ty)
                .collect::<Vec<_>>(),
            syn::Fields::Unit => todo!("Unit fields are not supported!"),
        },
        _ => todo!("AsVarType can only be derived for structs!"),
    };

    let crate_name = crate_name();

    let ident = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    quote! {
        impl #impl_generics #crate_name::vartype::AsVarType for #ident #ty_generics #where_clause{
            fn var_ty() -> &'static #crate_name::vartype::VarType{
                composite(&[
                    #(<#types as #crate_name::vartype::AsVarType>::var_ty()),*
                ])
            }
        }
    }
}
