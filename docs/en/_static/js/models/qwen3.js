// models/qwen3.js — Qwen3 model configuration for LMDeploy Config Generator
(function() {
    'use strict';
    window.LMDeployModelConfigs = window.LMDeployModelConfigs || {};

    window.LMDeployModelConfigs['qwen3'] = {
        name: 'Qwen3',

        dimensions: [
            {
                key: 'hardware', label: 'Hardware Platform', default: 'A100',
                options: [
                    { value: 'A100', label: 'A100(80G)' },
                    { value: 'H800', label: 'H800(80G)' },
                    { value: 'H200', label: 'H200(140G)' },
                    { value: 'V100', label: 'V100(32G)' }
                ]
            },
            {
                key: 'model_size', label: 'Model Size', default: '8B',
                options: [
                    { value: '235B-A22B', label: '235B MoE' },
                    { value: '30B-A3B',   label: '30B MoE' },
                    { value: '32B',       label: '32B' },
                    { value: '14B',       label: '14B' },
                    { value: '8B',        label: '8B' },
                    { value: '4B',        label: '4B' },
                    { value: '1.7B',      label: '1.7B' },
                    { value: '0.6B',      label: '0.6B' }
                ]
            },
            {
                key: 'quantization', label: 'Quantization', default: 'auto',
                options: [
                    { value: 'auto', label: 'Auto' },
                    { value: 'awq',  label: 'AWQ (W4A16)' },
                    { value: 'gptq', label: 'GPTQ (W4A16)' },
                    { value: 'fp8',  label: 'FP8' }
                ]
            },
            {
                key: 'category', label: 'Categories', default: 'instruct',
                options: [
                    { value: 'base',     label: 'Base' },
                    { value: 'instruct', label: 'Instruct' },
                    { value: 'thinking', label: 'Thinking' }
                ]
            },
            {
                key: 'reasoning_parser', label: 'Reasoning Parser', default: 'disabled',
                options: [
                    { value: 'disabled', label: 'Disabled' },
                    { value: 'enabled',  label: 'Enabled' }
                ]
            },
            {
                key: 'tool_call_parser', label: 'Tool Call Parser', default: 'disabled',
                options: [
                    { value: 'disabled', label: 'Disabled' },
                    { value: 'enabled',  label: 'Enabled' }
                ]
            }
        ],

        // GPU memory (GB) for TP estimation
        gpuMem: { 'A100': 80, 'H800': 80, 'H200': 140, 'V100': 32 },

        // Approximate BF16 model weight memory (GB)
        modelMem: {
            '235B-A22B': 440, '30B-A3B': 60, '32B': 64,
            '14B': 28, '8B': 16, '4B': 8, '1.7B': 4, '0.6B': 2
        },

        buildModelPath: function(sel) {
            var base = 'Qwen/Qwen3-' + sel.model_size;
            if (sel.category === 'instruct') base += '-Instruct';
            else if (sel.category === 'thinking') base += '-Thinking';
            return base;
        },

        buildExtraFlags: function(sel) {
            var flags = [];
            if (sel.quantization === 'awq') flags.push('--model-format awq');
            else if (sel.quantization === 'gptq') flags.push('--model-format gptq');
            if (sel.reasoning_parser === 'enabled') flags.push('--reasoning-parser qwen-qwq');
            if (sel.tool_call_parser === 'enabled') flags.push('--tool-call-parser qwen3');
            return flags;
        }
    };
})();
