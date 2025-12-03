// Modified from xgrammar/nanobind/nanobind.cc from xgrammar project.
/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/nanobind/nanobind.cc
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <xgrammar/xgrammar.h>

#include "src/turbomind/core/check.h"

namespace nb = nanobind;
using namespace xgrammar;

std::vector<std::string>
CommonEncodedVocabType(const nb::typed<nb::list, std::variant<std::string, nb::bytes>> encoded_vocab)
{
    std::vector<std::string> encoded_vocab_strs;
    encoded_vocab_strs.reserve(encoded_vocab.size());
    for (const auto& token : encoded_vocab) {
        if (nb::bytes result; nb::try_cast(token, result)) {
            encoded_vocab_strs.emplace_back(result.c_str());
        }
        else if (nb::str result; nb::try_cast(token, result)) {
            encoded_vocab_strs.emplace_back(result.c_str());
        }
        else {
            throw nb::type_error("Expected str or bytes for encoded_vocab");
        }
    }
    return encoded_vocab_strs;
}

std::vector<nanobind::bytes> TokenizerInfo_GetDecodedVocab(const TokenizerInfo& tokenizer)
{
    const auto&                  decoded_vocab = tokenizer.GetDecodedVocab();
    std::vector<nanobind::bytes> py_result;
    py_result.reserve(decoded_vocab.size());
    for (const auto& item : decoded_vocab) {
        py_result.emplace_back(nanobind::bytes(item.c_str()));
    }
    return py_result;
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

TokenizerInfo TokenizerInfo_DeserializeJSON(const std::string& json_string)
{
    auto result = TokenizerInfo::DeserializeJSON(json_string);
    if (std::holds_alternative<SerializationError>(result)) {
        throw std::get<SerializationError>(result);
    }
    return std::get<TokenizerInfo>(result);
}

NB_MODULE(_xgrammar, m)
{
    auto pyTokenizerInfo = nb::class_<TokenizerInfo>(m, "TokenizerInfo");
    pyTokenizerInfo
        .def(
            "__init__",
            [](TokenizerInfo*                                                  out,
               const nb::typed<nb::list, std::variant<std::string, nb::bytes>> encoded_vocab,
               int                                                             vocab_type,
               std::optional<int>                                              vocab_size,
               std::optional<std::vector<int32_t>>                             stop_token_ids,
               bool                                                            add_prefix_space) {
                new (out) TokenizerInfo{TokenizerInfo_Init(CommonEncodedVocabType(encoded_vocab),
                                                           vocab_type,
                                                           vocab_size,
                                                           std::move(stop_token_ids),
                                                           add_prefix_space)};
            },
            nb::arg("encoded_vocab"),
            nb::arg("vocab_type"),
            nb::arg("vocab_size").none(),
            nb::arg("stop_token_ids").none(),
            nb::arg("add_prefix_space"))
        .def_prop_ro("vocab_type", &TokenizerInfo_GetVocabType)
        .def_prop_ro("vocab_size", &TokenizerInfo::GetVocabSize)
        .def_prop_ro("add_prefix_space", &TokenizerInfo::GetAddPrefixSpace)
        .def_prop_ro("decoded_vocab", &TokenizerInfo_GetDecodedVocab)
        .def_prop_ro("stop_token_ids", &TokenizerInfo::GetStopTokenIds)
        .def_prop_ro("special_token_ids", &TokenizerInfo::GetSpecialTokenIds)
        .def("dump_metadata", &TokenizerInfo::DumpMetadata)
        .def_static("from_vocab_and_metadata",
                    [](const nb::typed<nb::list, std::variant<std::string, nb::bytes>> encoded_vocab,
                       const std::string&                                              metadata) {
                        return TokenizerInfo::FromVocabAndMetadata(CommonEncodedVocabType(encoded_vocab), metadata);
                    })
        .def_static("_detect_metadata_from_hf", &TokenizerInfo::DetectMetadataFromHF)
        .def("serialize_json", &TokenizerInfo::SerializeJSON)
        .def_static("deserialize_json", &TokenizerInfo_DeserializeJSON);

    auto pyCompiledGrammar = nb::class_<CompiledGrammar>(m, "CompiledGrammar");
    pyCompiledGrammar.def_prop_ro("grammar", &CompiledGrammar::GetGrammar)
        .def_prop_ro("tokenizer_info", &CompiledGrammar::GetTokenizerInfo)
        .def_prop_ro("memory_size_bytes", &CompiledGrammar::MemorySizeBytes)
        .def("serialize_json", &CompiledGrammar::SerializeJSON)
        .def_static("deserialize_json", &CompiledGrammar::DeserializeJSON);

    auto pyGrammarCompiler = nb::class_<GrammarCompiler>(m, "GrammarCompiler");
    pyGrammarCompiler.def(nb::init<const TokenizerInfo&, int, bool, int64_t>())
        .def("compile_json_schema",
             &GrammarCompiler::CompileJSONSchema,
             nb::call_guard<nb::gil_scoped_release>(),
             nb::arg("schema"),
             nb::arg("any_whitespace"),
             nb::arg("indent").none(),
             nb::arg("separators").none(),
             nb::arg("strict_mode"),
             nb::arg("max_whitespace_cnt").none())
        .def("compile_builtin_json_grammar",
             &GrammarCompiler::CompileBuiltinJSONGrammar,
             nb::call_guard<nb::gil_scoped_release>())
        .def("compile_regex", &GrammarCompiler::CompileRegex, nb::call_guard<nb::gil_scoped_release>())
        .def(
            "compile_grammar",
            [](GrammarCompiler& self, const Grammar& grammar) { return self.CompileGrammar(grammar); },
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "compile_grammar",
            [](GrammarCompiler& self, const std::string& ebnf_str, const std::string& root_rule_name) {
                return self.CompileGrammar(ebnf_str, root_rule_name);
            },
            nb::call_guard<nb::gil_scoped_release>())
        .def("clear_cache", &GrammarCompiler::ClearCache)
        .def("get_cache_size_bytes", &GrammarCompiler::GetCacheSizeBytes)
        .def_prop_ro("cache_limit_bytes", &GrammarCompiler::CacheLimitBytes);
}
