use proc_macro2::TokenStream;
use quote::quote;

use crate::utils::crate_name;

pub fn record_impl(closure: syn::ExprClosure) -> TokenStream {
    let crate_name = crate_name();

    let inputs = &closure.inputs;
    let (input_vars, input_types): (Vec<_>, Vec<_>) = closure
        .inputs
        .iter()
        .map(|arg| match &arg {
            syn::Pat::Type(arg) => {
                let pat = &arg.pat;
                let ty = &arg.ty;
                (quote!(#pat), quote!(#ty))
            }
            _ => todo!(),
        })
        .unzip();

    let lazy = quote!(#crate_name::once_cell::sync::Lazy);
    let input_tuple_vars = quote!((#(#input_vars,)*));

    quote! {
        move |device: &#crate_name::Device, #inputs|{

            let func = #closure;

            use std::sync::Mutex;
            use #crate_name::record::FCache;
            use #crate_name::record::WrapInput;
            static FCACHE: #lazy<Mutex<FCache>> = #lazy::new(||Mutex::new(FCache::default()));

            let wrapped_input_func = func.wrap_input();

            FCACHE.lock().unwrap().call(wrapped_input_func, device, #input_tuple_vars)
        }
    }
}
