use crate::trace::Trace;

extern crate pest;

use pest::Parser;

#[derive(Parser)]
#[grammar = "grammars/python.pest"]
struct PythonParser;

pub fn parse(source: &str) -> Trace {
    todo!()
}

#[cfg(test)]
mod test {
    use super::{PythonParser, Rule};
    use pest::Parser;

    #[test]
    fn basic_python() {
        let t = PythonParser::parse(Rule::function_call_or_sig, "def test(a)");
        dbg!(&t);
    }
}
