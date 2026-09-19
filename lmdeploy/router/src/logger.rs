use chrono::{Datelike, Timelike};
use log::{Level, Metadata, Record};
use nix;
pub struct Logger;

impl Logger {
    fn thread_id() -> i32 {
        // TODO Use gettid() on Linux
        nix::unistd::getpid().into()
    }

    fn to_char(level: Level) -> char {
        match level {
            Level::Error => 'E',
            Level::Warn => 'W',
            Level::Info => 'I',
            Level::Debug => 'D',
            Level::Trace => 'T',
        }
    }

    fn filename(filename: &str) -> &str {
        filename.rsplit_once('/').unwrap().1
    }
}

impl log::Log for Logger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= Level::Info
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }

        let now = chrono::Local::now();

        println!(
            "{}{:02}{:02} {:02}:{:02}:{:02}.{:06} {} {}:{}] {}",
            Logger::to_char(record.level()),
            now.month(),
            now.day(),
            now.hour(),
            now.minute(),
            now.second(),
            now.timestamp_subsec_micros(),
            Logger::thread_id(),
            Logger::filename(record.file().unwrap_or("<unknown>")),
            record.line().unwrap_or(0),
            record.args()
        );
    }

    fn flush(&self) {}
}
