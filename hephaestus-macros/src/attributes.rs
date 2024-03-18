use crate::utils::crate_name;
use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::punctuated::Punctuated;
use syn::{parse_quote, FnArg, Ident, Signature, Token};

pub fn record_impl(func: syn::ItemFn) -> TokenStream {
    let crate_name = crate_name();

    let ident = &func.sig.ident;
    let input = &func.sig.inputs;
    let output = &func.sig.output;

    let input_vars = input
        .iter()
        .map(|arg| match arg {
            syn::FnArg::Receiver(_) => {
                todo!("The recorded macro is not implemented for member functions!")
            }
            syn::FnArg::Typed(arg) => {
                let pat = &arg.pat;
                quote!(#pat)
            }
        })
        .collect::<Vec<_>>();
    let input_types = input
        .iter()
        .map(|arg| match arg {
            syn::FnArg::Receiver(_) => quote!(Self),
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

    let mut func = func.clone();
    let func_inputs: Punctuated<FnArg, Token![,]> = parse_quote!(#(#input_vars: #input_types,)*);
    let func_ident = Ident::new(
        &format!("_{ident}", ident = &func.sig.ident),
        Span::call_site(),
    );
    func.sig.ident = func_ident.clone();
    func.sig.inputs = func_inputs;

    quote! {
        fn #ident(device: &#crate_name::Device, #input) -> #output_type{
            use #crate_name::record::Recordable;
            #func
            //
            const RECORDING: #lazy<#crate_name::record::Func<#input_tuple_type, #output_type>> = #lazy::new(|| #func_ident.func());
            RECORDING.call(device, (#(#input_vars,)*))
        }
    }
}
