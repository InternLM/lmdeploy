// models/internlm.js — InternLM model configuration for LMDeploy Config Generator
(function() {
    'use strict';
    window.LMDeployModelConfigs = window.LMDeployModelConfigs || {};

    window.LMDeployModelConfigs['internlm'] = {
        name: 'InternLM',

        dimensions: [
            {
                key: 'hardware', label: 'Hardware Platform', default: 'A100',
                options: [
                    { value: 'A100', label: 'A100(80G)' },
                    { value: 'H800', label: 'H800(80G)' },
                    { value: 'H200', label: 'H200(140G)' }
                ]
            },
            {
                key: 'model_size', label: 'Model Version', default: 'InternLM3-8B',
                options: [
                    { value: 'InternLM2-7B',   label: 'InternLM2 (7B)' },
                    { value: 'InternLM2-20B',  label: 'InternLM2 (20B)' },
                    { value: 'InternLM2.5-7B', label: 'InternLM2.5 (7B)' },
                    { value: 'InternLM3-8B',   label: 'InternLM3 (8B)' }
                ]
            },
            {
                key: 'quantization', label: 'Quantization', default: 'auto',
                options: [
                    { value: 'auto', label: 'Auto (BF16)' },
                    { value: 'awq',  label: 'AWQ (W4A16)' },
                    { value: 'kv8',  label: 'KV Cache INT8' }
                ]
            },
            {
                key: 'category', label: 'Categories', default: 'chat',
                options: [
                    { value: 'base', label: 'Base' },
                    { value: 'chat', label: 'Chat' }
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

        gpuMem: { 'A100': 80, 'H800': 80, 'H200': 140 },

        modelMem: {
            'InternLM2-7B': 14, 'InternLM2-20B': 40,
            'InternLM2.5-7B': 14, 'InternLM3-8B': 16
        },

        buildModelPath: function(sel) {
            var chatMap = {
                'InternLM2-7B':   'internlm/internlm2-chat-7b',
                'InternLM2-20B':  'internlm/internlm2-chat-20b',
                'InternLM2.5-7B': 'internlm/internlm2_5-7b-chat',
                'InternLM3-8B':   'internlm/internlm3-8b-instruct'
            };
            var baseMap = {
                'InternLM2-7B':   'internlm/internlm2-7b',
                'InternLM2-20B':  'internlm/internlm2-20b',
                'InternLM2.5-7B': 'internlm/internlm2_5-7b',
                'InternLM3-8B':   'internlm/internlm3-8b'
            };
            if (sel.category === 'base') {
                return baseMap[sel.model_size] || 'internlm/internlm3-8b';
            }
            return chatMap[sel.model_size] || 'internlm/internlm3-8b-instruct';
        },

        buildExtraFlags: function(sel) {
            var flags = [];
            if (sel.quantization === 'awq') flags.push('--model-format awq');
            else if (sel.quantization === 'kv8') flags.push('--quant-policy 8');
            if (sel.reasoning_parser === 'enabled') flags.push('--reasoning-parser intern-s1');
            if (sel.tool_call_parser === 'enabled') flags.push('--tool-call-parser internlm');
            return flags;
        }
    };
})();
