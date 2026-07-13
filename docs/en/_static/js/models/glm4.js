// models/glm4.js — GLM-4 model configuration for LMDeploy Config Generator
(function() {
    'use strict';
    window.LMDeployModelConfigs = window.LMDeployModelConfigs || {};

    window.LMDeployModelConfigs['glm4'] = {
        name: 'GLM-4',

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
                key: 'model_size', label: 'Model Version', default: 'GLM-4-9B',
                options: [
                    { value: 'GLM-4-9B',          label: 'GLM-4 (9B)' },
                    { value: 'GLM-4-0414-9B',     label: 'GLM-4-0414 (9B)' },
                    { value: 'GLM-4.5-355B',      label: 'GLM-4.5 (355B)' },
                    { value: 'GLM-4.5-Air-106B',  label: 'GLM-4.5-Air (106B)' },
                    { value: 'GLM-4.7-Flash-30B', label: 'GLM-4.7-Flash (30B)' },
                    { value: 'GLM-5-754B',        label: 'GLM-5 (754B)' }
                ]
            },
            {
                key: 'quantization', label: 'Quantization', default: 'auto',
                options: [
                    { value: 'auto', label: 'Auto (BF16)' },
                    { value: 'awq',  label: 'AWQ (W4A16)' }
                ]
            },
            {
                key: 'category', label: 'Categories', default: 'chat',
                options: [
                    { value: 'chat', label: 'Chat' }
                ]
            }
        ],

        gpuMem: { 'A100': 80, 'H800': 80, 'H200': 140 },

        modelMem: {
            'GLM-4-9B': 18, 'GLM-4-0414-9B': 18,
            'GLM-4.5-355B': 700, 'GLM-4.5-Air-106B': 212,
            'GLM-4.7-Flash-30B': 60, 'GLM-5-754B': 1400
        },

        buildModelPath: function(sel) {
            var map = {
                'GLM-4-9B':          'THUDM/glm-4-9b-chat',
                'GLM-4-0414-9B':     'THUDM/GLM-4-0414-9B-Chat',
                'GLM-4.5-355B':      'THUDM/GLM-4.5-355B-Chat',
                'GLM-4.5-Air-106B':  'THUDM/GLM-4.5-Air-106B-Chat',
                'GLM-4.7-Flash-30B': 'THUDM/GLM-4.7-Flash-30B',
                'GLM-5-754B':        'THUDM/GLM-5-754B'
            };
            return map[sel.model_size] || 'THUDM/glm-4-9b-chat';
        },

        buildExtraFlags: function(sel) {
            var flags = [];
            if (sel.quantization === 'awq') flags.push('--model-format awq');
            return flags;
        }
    };
})();
