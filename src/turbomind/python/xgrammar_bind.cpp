// Modified from xgrammar/nanobind/nanobind.cc from xgrammar project.
/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/nanobind/nanobind.cc
 */

#include <memory>
#include <sstream>
#include <stdexcept>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <xgrammar/xgrammar.h>

#include "src/turbomind/core/check.h"

namespace py = pybind11;
using namespace xgrammar;
using namespace pybind11::literals;

namespace {

static const std::vector<std::string>
CommonEncodedVocabType(const py::typing::List<std::variant<std::string, py::bytes>>& lst)
{
    std::vector<std::string> out;
    out.reserve(lst.size());
    for (const auto& h : lst) {
        if (py::isinstance<py::str>(h)) {
            out.emplace_back(h.cast<std::string>());
        }
        else if (py::isinstance<py::bytes>(h)) {
            out.emplace_back(h.cast<py::bytes>());
        }
        else {
            throw std::invalid_argument("encoded_vocab items must be str or bytes");
        }
    }
    return out;
}

TokenizerInfo TokenizerInfo_Init(const std::vector<std::string>&     encoded_vocab,
                                 int                                 vocab_type,
                                 std::optional<int>                  vocab_size,
                                 std::optional<std::vector<int32_t>> stop_token_ids,
                                 bool                                add_prefix_space)
{
    TM_CHECK(vocab_type == 0 || vocab_type == 1 || vocab_type == 2) << "Invalid vocab type: " << vocab_type;
    return TokenizerInfo(
        encoded_vocab, static_cast<VocabType>(vocab_type), vocab_size, stop_token_ids, add_prefix_space);
}

int TokenizerInfo_GetVocabType(const TokenizerInfo& tokenizer)
{
    return static_cast<int>(tokenizer.GetVocabType());
}

std::vector<py::bytes> TokenizerInfo_GetDecodedVocab(const TokenizerInfo& tokenizer)
{
    const auto&            decoded_vocab = tokenizer.GetDecodedVocab();
    std::vector<py::bytes> py_result;
    py_result.reserve(decoded_vocab.size());
    for (const auto& item : decoded_vocab) {
        py_result.emplace_back(py::bytes(item.c_str()));
    }
    return py_result;
}

}  // namespace

PYBIND11_MODULE(_xgrammar, m)
{
    py::class_<TokenizerInfo, std::shared_ptr<TokenizerInfo>>(m, "TokenizerInfo")
        .def(py::init([](const py::typing::List<std::variant<std::string, py::bytes>>& encoded_vocab,
                         int                                                           vocab_type,
                         std::optional<int>                                            vocab_size,
                         std::optional<std::vector<int32_t>>                           stop_token_ids,
                         bool                                                          add_prefix_space) {
                 return TokenizerInfo{TokenizerInfo_Init(CommonEncodedVocabType(encoded_vocab),
                                                         vocab_type,
                                                         vocab_size,
                                                         std::move(stop_token_ids),
                                                         add_prefix_space)};
             }),
             py::arg("encoded_vocab"),
             py::arg("vocab_type"),
             py::arg("vocab_size")     = py::none(),
             py::arg("stop_token_ids") = py::none(),
             py::arg("add_prefix_space"))

        .def_property_readonly("vocab_type", &TokenizerInfo_GetVocabType)
        .def_property_readonly("vocab_size", &TokenizerInfo::GetVocabSize)
        .def_property_readonly("add_prefix_space", &TokenizerInfo::GetAddPrefixSpace)
        .def_property_readonly("decoded_vocab", &TokenizerInfo_GetDecodedVocab)
        .def_property_readonly("stop_token_ids", &TokenizerInfo::GetStopTokenIds)
        .def_property_readonly("special_token_ids", &TokenizerInfo::GetSpecialTokenIds)

        .def("dump_metadata", &TokenizerInfo::DumpMetadata)

        .def_static("from_vocab_and_metadata",
                    [](const py::typing::List<std::variant<std::string, py::bytes>>& encoded_vocab,
                       const std::string&                                            metadata) {
                        return TokenizerInfo::FromVocabAndMetadata(CommonEncodedVocabType(encoded_vocab), metadata);
                    })

        .def_static("_detect_metadata_from_hf", &TokenizerInfo::DetectMetadataFromHF);

    py::class_<CompiledGrammar>(m, "CompiledGrammar");

    py::class_<GrammarCompiler> pyGrammarCompiler(m, "GrammarCompiler");
    pyGrammarCompiler
        .def(py::init<const TokenizerInfo&, int, bool, int64_t>(),
             py::arg("tokenizer_info"),
             py::arg("max_threads")      = 8,
             py::arg("cache_enabled")    = true,
             py::arg("max_memory_bytes") = -1)
        .def("compile_json_schema",
             &GrammarCompiler::CompileJSONSchema,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("schema"),
             py::arg("any_whitespace")     = false,
             py::arg("indent")             = py::none(),
             py::arg("separators")         = py::none(),
             py::arg("strict_mode")        = true,
             py::arg("max_whitespace_cnt") = py::none())
        .def("compile_regex",
             &GrammarCompiler::CompileRegex,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("schema"));
}
