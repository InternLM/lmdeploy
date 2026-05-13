// models/deepseek.js — DeepSeek model configuration for LMDeploy Config Generator
(function() {
    'use strict';
    window.LMDeployModelConfigs = window.LMDeployModelConfigs || {};

    window.LMDeployModelConfigs['deepseek'] = {
        name: 'DeepSeek',

        dimensions: [
            {
                key: 'hardware', label: 'Hardware Platform', default: 'H800',
                options: [
                    { value: 'A100', label: 'A100(80G)' },
                    { value: 'H800', label: 'H800(80G)' },
                    { value: 'H200', label: 'H200(140G)' }
                ]
            },
            {
                key: 'model_size', label: 'Model Version', default: 'V3',
                options: [
                    { value: 'V2-Lite', label: 'V2 Lite (16B)' },
                    { value: 'V2',      label: 'V2 (236B)' },
                    { value: 'V2.5',    label: 'V2.5 (236B)' },
                    { value: 'V3',      label: 'V3 (685B)' },
                    { value: 'V3.2',    label: 'V3.2 (685B)' }
                ]
            },
            {
                key: 'quantization', label: 'Quantization', default: 'auto',
                options: [
                    { value: 'auto', label: 'Auto (BF16)' }
                ]
            },
            {
                key: 'reasoning_parser', label: 'Reasoning Parser', default: 'disabled',
                options: [
                    { value: 'disabled', label: 'Disabled' },
                    { value: 'enabled',  label: 'Enabled' }
                ]
            }
        ],

        gpuMem: { 'A100': 80, 'H800': 80, 'H200': 140 },

        modelMem: {
            'V2-Lite': 32, 'V2': 440, 'V2.5': 440,
            'V3': 1300, 'V3.2': 1300
        },

        buildModelPath: function(sel) {
            var map = {
                'V2-Lite': 'deepseek-ai/DeepSeek-V2-Lite-Chat',
                'V2':      'deepseek-ai/DeepSeek-V2-Chat',
                'V2.5':    'deepseek-ai/DeepSeek-V2.5',
                'V3':      'deepseek-ai/DeepSeek-V3',
                'V3.2':    'deepseek-ai/DeepSeek-V3-0324'
            };
            return map[sel.model_size] || 'deepseek-ai/DeepSeek-V3';
        },

        buildExtraFlags: function(sel) {
            var flags = [];
            flags.push('--backend pytorch');
            if (sel.reasoning_parser === 'enabled') flags.push('--reasoning-parser deepseek-r1');
            return flags;
        }
    };
})();
