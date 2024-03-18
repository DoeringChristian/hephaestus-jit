use crate::utils::crate_name;
use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

pub fn record_impl(func: syn::ItemFn) -> TokenStream {
    let crate_name = crate_name();

    let func_ident = &func.sig.ident;
    let input = &func.sig.inputs;
    let output = &func.sig.output;

    let input_vars = input
        .iter()
        .map(|arg| match arg {
            syn::FnArg::Receiver(_) => todo!(),
            syn::FnArg::Typed(arg) => {
                let pat = &arg.pat;
                quote!(#pat)
            }
        })
        .collect::<Vec<_>>();
    let input_types = input
        .iter()
        .map(|arg| match arg {
            syn::FnArg::Receiver(_) => todo!(),
            syn::FnArg::Typed(arg) => {
                let ty = &arg.ty;
                quote!(#ty)
            }
        })
        .collect::<Vec<_>>();
    let output_type = match output {
        syn::ReturnType::Default => quote!(()),
        syn::ReturnType::Type(_, ty) => quote!(#ty),
    };

    let lazy = quote!(#crate_name::once_cell::sync::Lazy);
    let input_tuple_type = quote!((#(#input_types,)*));

    quote! {
        fn #func_ident(device: &#crate_name::Device, #input) -> #output_type{
            #func
            //
            const RECORDING: #lazy<#crate_name::record::Func<#input_tuple_type, #output_type>> = #lazy::new(|| #func_ident.func());
            RECORDING.call(device, (#(#input_vars,)*))
        }
    }
}
