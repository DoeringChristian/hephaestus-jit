use crate::utils::jit_name;
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

    let jit = jit_name();

    let ident = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    quote! {
        impl #impl_generics #jit::vartype::AsVarType for #ident #ty_generics #where_clause{
            fn var_ty() -> &'static #jit::vartype::VarType{
                #jit::vartype::composite(&[
                    #(<#types as #jit::vartype::AsVarType>::var_ty()),*
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

    let jit = jit_name();

    let n_names = names.len();

    let ident = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    quote! {
        impl #impl_generics #jit::Traverse for #ident #ty_generics #where_clause{
            fn traverse(&self, vars: &mut Vec<#jit::VarRef>) -> &'static #jit::Layout{
                let layouts = [#(self.#names.traverse(vars)),*];
                #jit::Layout::tuple(&layouts)
            }
            fn ravel(&self) -> #jit::VarRef {
                let refs = [#(self.#names.ravel()),*];
                #jit::composite(&refs)
            }
            fn hash(&self, state: &mut dyn std::hash::Hasher) {
                #(self.#names.hash(state);)*
            }
        }
    }
}

pub fn derive_construct_impl(input: DeriveInput) -> TokenStream {
    let jit = jit_name();

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
                    impl #impl_generics #jit::Construct for #ident #ty_generics #where_clause{
                        fn construct(vars: &mut impl Iterator<Item = #jit::VarRef>, layout: &'static #jit::Layout) -> Self{
                            let mut layouts = layout.tuple_types().unwrap().into_iter();
                            Self{
                                #(#names: <#types as #jit::Construct>::construct(vars, layouts.next().unwrap()),)*
                            }
                        }
                        fn unravel(var: impl Into<#jit::VarRef>) -> Self{
                            let var = var.into();
                            let ty = var.ty();
                            assert!(matches!(ty, #jit::vartype::VarType::Struct{..}));
                            let mut iter = var.extract_all();
                            Self{
                                #(
                                    #names: <#types as #jit::Construct>::unravel(iter.next().unwrap()),
                                )*
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
                    impl #impl_generics #jit::Construct for #ident #ty_generics #where_clause{
                        fn construct(vars: &mut impl Iterator<Item = #jit::VarRef>, layout: &'static #jit::Layout) -> Self{
                            let mut layouts = layout.tuple_types().unwrap().into_iter();
                            Self(#(<#types as #jit::Construct>::construct(vars, layouts.next().unwrap()),)*)
                        }
                        fn unravel(var: impl Into<#jit::VarRef>) -> Self{
                            let var = var.into();
                            let ty = var.ty();
                            assert!(matches!(ty, #jit::vartype::VarType::Struct{..}));
                            let mut iter = var.extract_all();
                            Self(#(<#types as #jit::Construct>::unravel(iter.next().unwrap()),)*)
                        }
                    }
                }
            }
            syn::Fields::Unit => todo!("Unit fields are not supported!"),
        },
        _ => todo!("Construct can only be derived for structs!"),
    }
}
