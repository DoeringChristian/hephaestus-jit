use proc_macro2::{Span, TokenStream, TokenTree};
use quote::quote;
use syn::token::Token;
use syn::{DeriveInput, Ident, Lifetime, LitInt};

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

pub fn derive_traverse_impl(input: DeriveInput) -> TokenStream {
    // Get the types of the struct
    let names = match &input.data {
        syn::Data::Struct(data) => match &data.fields {
            syn::Fields::Named(fields) => fields
                .named
                .iter()
                .map(|field| {
                    let ident = field.ident.as_ref().unwrap();
                    quote!(#ident)
                })
                .collect::<Vec<_>>(),
            syn::Fields::Unnamed(fields) => fields
                .unnamed
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    // let ident = Ident::new(&format!("{i}"), Span::call_site());
                    let ident = format!("{i}");
                    let ident = LitInt::new(&ident, Span::call_site());
                    quote!(#ident)
                })
                .collect::<Vec<_>>(),
            syn::Fields::Unit => todo!("Unit fields are not supported!"),
        },
        _ => todo!("Traverse can only be derived for structs!"),
    };

    let crate_name = crate_name();

    let ident = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    // Construct a livetime that doesn't exist
    let livetime = input
        .generics
        .params
        .iter()
        .filter_map(|param| match param {
            syn::GenericParam::Lifetime(lt) => Some(lt.lifetime.ident.to_string()),
            _ => None,
        })
        .fold(String::from("'lt_"), |mut a, b| {
            a.push_str(&b);
            a.push_str("_");
            a
        });
    let livetime = Lifetime::new(&livetime, Span::call_site());

    quote! {
        impl #impl_generics #crate_name::Traverse for #ident #ty_generics #where_clause{
            fn traverse<#livetime>(&#livetime self, vec: &mut Vec<&#livetime VarRef>){
                #(
                    self.#names.traverse(vec);
                )*
            }
        }
    }
}

pub fn derive_construct_impl(input: DeriveInput) -> TokenStream {
    let crate_name = crate_name();

    let ident = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    match input.data {
        syn::Data::Struct(data) => match &data.fields {
            syn::Fields::Named(fields) => {
                let names = fields
                    .named
                    .iter()
                    .map(|field| field.ident.as_ref().unwrap())
                    .collect::<Vec<_>>();
                let types = fields
                    .named
                    .iter()
                    .map(|field| &field.ty)
                    .collect::<Vec<_>>();
                quote! {
                    impl #impl_generics #crate_name::Construct for #ident #ty_generics #where_clause{
                        fn construct(iter: &mut impl Iterator<Item = #crate_name::VarRef>) -> Self{
                            Self{
                                #(#names: <#types as #crate_name::Construct>::construct(iter),)*
                            }
                        }
                    }
                }
            }
            syn::Fields::Unnamed(fields) => {
                let types = fields
                    .unnamed
                    .iter()
                    .map(|field| &field.ty)
                    .collect::<Vec<_>>();
                quote! {
                    impl #impl_generics #crate_name::Construct for #ident #ty_generics #where_clause{
                        fn construct(iter: &mut impl Iterator<Item = #crate_name::VarRef>) -> Self{
                            Self(#(<#types as #crate_name::Construct>::construct(iter),)*)
                        }
                    }
                }
            }
            syn::Fields::Unit => todo!("Unit fields are not supported!"),
        },
        _ => todo!("Construct can only be derived for structs!"),
    }
}
