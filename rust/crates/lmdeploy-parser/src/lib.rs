// Copyright (c) OpenMMLab. All rights reserved.

//! Request-scoped parsers for reasoning and tool-call output.

mod response;

pub use response::{AssistantEvent, ParserConfig, ResponseParser, SUPPORTED_TOOL_PARSERS};
