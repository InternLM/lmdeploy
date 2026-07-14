# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Which versions are eligible for receiving such patches depend on the severity of the vulnerability and the age of the version.

| Version | Supported          |
| ------- | ------------------ |
| 0.13.x  | :white_check_mark: |
| 0.12.x  | :white_check_mark: |
| 0.11.x  | :x:                |
| < 0.11  | :x:                |

## Reporting a Vulnerability

We take security issues seriously. We appreciate your efforts to responsibly disclose your findings, and will make every effort to acknowledge your contributions.

If you believe you have found a security vulnerability in LMDeploy, please report it to us by sending an email to openmmlab@gmail.com. Please include the following details:

- Description of the vulnerability
- Steps to reproduce the issue
- Impact of the vulnerability
- Any proposed fix or mitigation

We will review your report and respond within 48 hours. If the issue is confirmed, we will:

1. Acknowledge the vulnerability
2. Work on a fix
3. Release a security update
4. Credit you in the release notes (if you wish)

Please do not disclose the vulnerability publicly until we have had a chance to address it.

## Security Best Practices

When using LMDeploy, please follow these security best practices:

1. **Keep LMDeploy updated**: Regularly update to the latest version to benefit from security fixes.
2. **Use trusted models**: Only load models from trusted sources. Set `trust_remote_code=False` unless you explicitly trust the model provider.
3. **Secure your deployment**: If serving models via API, ensure proper authentication and authorization mechanisms are in place.
4. **Monitor for suspicious activity**: Keep an eye on logs and metrics for any unusual patterns.
5. **Limit exposure**: Do not expose LMDeploy services directly to the internet without proper security measures.

## Known Security Considerations

### Remote Code Execution Risk

Loading models with `trust_remote_code=True` can execute arbitrary code from the model repository. Only use this option with models from trusted sources.

### Model Input Sanitization

LMDeploy does not sanitize user inputs by default. If you are building a public-facing service, you should implement input validation and sanitization to prevent injection attacks.

### GPU Memory Safety

Large model inputs can exhaust GPU memory, potentially causing denial of service. Consider implementing input size limits.
