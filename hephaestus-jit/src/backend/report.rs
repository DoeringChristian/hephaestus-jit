#[derive(Debug)]
pub struct PassReport {
    pub name: String,
    pub start: std::time::Duration, // duration since start of frame
    pub duration: std::time::Duration,
}

#[derive(Debug, Default)]
pub struct ExecReport {
    pub cpu_start: Option<std::time::SystemTime>,
    pub cpu_duration: std::time::Duration,
    pub passes: Vec<PassReport>,
}

#[derive(Default)]
pub struct Report {
    pub exec: ExecReport,
}
impl std::fmt::Debug for Report {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Passes:")?;
        for pass in self.exec.passes.iter() {
            writeln!(
                f,
                "\t{name: <50} {duration:?} @ {start:?}",
                name = pass.name,
                duration = pass.duration,
                start = pass.start,
            )?;
        }
        Ok(())
    }
}
impl Report {
    pub fn submit_to_profiler(&self) {
        #[cfg(feature = "profile-with-puffin")]
        use profiling::puffin;
        let start_ns = self
            .exec
            .cpu_start
            .and_then(|start| {
                Some(start.duration_since(std::time::UNIX_EPOCH).ok()?.as_nanos() as i64)
            })
            .unwrap_or(puffin::now_ns().into());

        // let start_ns = puffin::now_ns();
        let mut stream = puffin::Stream::default();

        let scope_details = self
            .exec
            .passes
            .iter()
            .map(|pass| puffin::ScopeDetails::from_scope_name(pass.name.clone()))
            .collect::<Vec<_>>();

        let ids = puffin::GlobalProfiler::lock().register_user_scopes(&scope_details);

        for (pass, id) in self.exec.passes.iter().zip(ids.into_iter()) {
            let start = stream.begin_scope(|| start_ns + pass.start.as_nanos() as i64, id, "");
            stream.end_scope(
                start.0,
                start_ns + (pass.start.as_nanos() + pass.duration.as_nanos()) as i64,
            );
        }
        // let stream_info = puffin::StreamInfo::parse(stream).unwrap();
        puffin::GlobalProfiler::lock().report_user_scopes(
            puffin::ThreadInfo {
                start_time_ns: None,
                name: "gpu".into(),
            },
            &puffin::StreamInfo {
                stream,
                num_scopes: 0,
                depth: 1,
                range_ns: (
                    start_ns,
                    start_ns
                        + (self.exec.passes.last().unwrap().start.as_nanos()
                            + self.exec.passes.last().unwrap().duration.as_nanos())
                            as i64,
                ),
            }
            .as_stream_into_ref(), //& stream_info.as_stream_into_ref(),
        );
    }
}
