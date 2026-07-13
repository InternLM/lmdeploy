// LMDeploy Interactive Configuration Generator — Generic Engine
// Model-specific configurations are loaded from js/models/*.js via
// the window.LMDeployModelConfigs global registry.
(function() {
    'use strict';

    function initConfigGenerator() {
        var container = document.getElementById('lmdeploy-config-generator');
        if (!container) return;

        // ── Read model config from registry ──────────────────────────
        var configKey = container.getAttribute('data-model-config') || 'qwen3';
        var configs = window.LMDeployModelConfigs || {};
        var config = configs[configKey];
        if (!config) {
            container.textContent = 'Unknown model config: ' + configKey +
                '. Available: ' + Object.keys(configs).join(', ');
            return;
        }

        // ── TP estimation (generic) ─────────────────────────────────
        function getRecommendedTP(sel) {
            var mem = (config.gpuMem || {})[sel.hardware] || 80;
            var need = (config.modelMem || {})[sel.model_size] || 16;
            if (sel.quantization === 'awq' || sel.quantization === 'gptq') {
                need *= 0.3;
            } else if (sel.quantization === 'fp8') {
                need *= 0.55;
            }
            var tp = 1;
            while (tp * mem < need * 1.15 && tp < 8) {
                tp *= 2;
            }
            return tp;
        }

        // ── Generate command ────────────────────────────────────────
        function generateCommand() {
            var sel = {};
            container.querySelectorAll('.cg-pill-bar').forEach(function(bar) {
                var key = bar.getAttribute('data-key');
                var active = bar.querySelector('.cg-pill.active');
                if (active) sel[key] = active.getAttribute('data-value');
            });

            var modelPath = config.buildModelPath(sel);
            var tp = getRecommendedTP(sel);
            var parts = ['lmdeploy serve api_server ' + modelPath];

            if (tp > 1) parts.push('--tp ' + tp);

            var extraFlags = config.buildExtraFlags ? config.buildExtraFlags(sel) : [];
            parts = parts.concat(extraFlags);

            if (parts.length <= 2) return parts.join(' ');
            return parts[0] + ' \\\n' +
                parts.slice(1).map(function(p) { return '  ' + p; }).join(' \\\n');
        }

        // ── Update command display ──────────────────────────────────
        function updateCommand() {
            var el = container.querySelector('.cg-generated-command');
            if (el) el.textContent = generateCommand();
        }

        // ── Render a single dimension row ───────────────────────────
        function renderDimension(dim) {
            var row = document.createElement('div');
            row.className = 'cg-row';

            var label = document.createElement('div');
            label.className = 'cg-label';
            label.textContent = dim.label;
            row.appendChild(label);

            var bar = document.createElement('div');
            bar.className = 'cg-pill-bar';
            bar.setAttribute('data-key', dim.key);

            dim.options.forEach(function(opt) {
                var pill = document.createElement('button');
                pill.className = 'cg-pill';
                pill.setAttribute('data-value', opt.value);
                pill.textContent = opt.label;
                if (opt.value === dim.default) pill.classList.add('active');

                pill.addEventListener('click', function() {
                    bar.querySelectorAll('.cg-pill').forEach(function(p) {
                        p.classList.remove('active');
                    });
                    pill.classList.add('active');
                    updateCommand();
                });

                bar.appendChild(pill);
            });

            row.appendChild(bar);
            return row;
        }

        // ── Build the full UI ───────────────────────────────────────
        var wrapper = document.createElement('div');
        wrapper.className = 'cg-wrapper';

        config.dimensions.forEach(function(dim) {
            wrapper.appendChild(renderDimension(dim));
        });

        // Command output section
        var cmdSection = document.createElement('div');
        cmdSection.className = 'cg-command-section';

        var cmdLabel = document.createElement('div');
        cmdLabel.className = 'cg-command-label';
        cmdLabel.textContent = 'Generated Command';
        cmdSection.appendChild(cmdLabel);

        var cmdBox = document.createElement('div');
        cmdBox.className = 'cg-command-box';

        var pre = document.createElement('pre');
        var code = document.createElement('code');
        code.className = 'cg-generated-command';
        pre.appendChild(code);
        cmdBox.appendChild(pre);

        var copyBtn = document.createElement('button');
        copyBtn.className = 'cg-copy-btn';
        copyBtn.textContent = 'Copy';
        copyBtn.addEventListener('click', function() {
            var text = code.textContent;
            navigator.clipboard.writeText(text).then(function() {
                copyBtn.textContent = 'Copied!';
                setTimeout(function() { copyBtn.textContent = 'Copy'; }, 2000);
            }).catch(function() {
                // Fallback for older browsers
                var ta = document.createElement('textarea');
                ta.value = text;
                ta.style.position = 'fixed';
                ta.style.left = '-9999px';
                document.body.appendChild(ta);
                ta.select();
                document.execCommand('copy');
                document.body.removeChild(ta);
                copyBtn.textContent = 'Copied!';
                setTimeout(function() { copyBtn.textContent = 'Copy'; }, 2000);
            });
        });
        cmdBox.appendChild(copyBtn);

        cmdSection.appendChild(cmdBox);
        wrapper.appendChild(cmdSection);

        container.appendChild(wrapper);
        updateCommand();
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initConfigGenerator);
    } else {
        initConfigGenerator();
    }
})();
