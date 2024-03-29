use crate::utils::crate_name;
use proc_macro2::{Span, TokenStream, TokenTree};
use quote::quote;
use syn::token::Token;
use syn::{DeriveInput, Ident, Lifetime, LitInt};

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
                #crate_name::vartype::composite(&[
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

    let n_names = names.len();

    let ident = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    quote! {
        impl #impl_generics #crate_name::Traverse for #ident #ty_generics #where_clause{
            fn traverse(&self, vars: &mut Vec<VarRef>, layout: &mut Vec<usize>){
                layout.push(#n_names);
                #(
                    self.#names.traverse(vars, layout);
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
                let n_params = types.len();
                quote! {
                    impl #impl_generics #crate_name::Construct for #ident #ty_generics #where_clause{
                        fn construct(vars: &mut impl Iterator<Item = #crate_name::VarRef>, layout: &mut impl Iterator<Item = usize>) -> Self{
                            assert_eq!(layout.next().unwrap(), #n_params);
                            Self{
                                #(#names: <#types as #crate_name::Construct>::construct(vars, layout),)*
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
                let n_params = types.len();
                quote! {
                    impl #impl_generics #crate_name::Construct for #ident #ty_generics #where_clause{
                        fn construct(vars: &mut impl Iterator<Item = #crate_name::VarRef>, layout: &mut impl Iterator<Item = usize>) -> Self{
                            assert_eq!(layout.next().unwrap(), #n_params);
                            Self(#(<#types as #crate_name::Construct>::construct(vars, layout),)*)
                        }
                    }
                }
            }
            syn::Fields::Unit => todo!("Unit fields are not supported!"),
        },
        _ => todo!("Construct can only be derived for structs!"),
    }
}
