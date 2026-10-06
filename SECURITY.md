# Security Policy

## Supported Versions

DevWhisper is currently in active development. Security updates are provided for the latest version on the `main` branch.

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of DevWhisper seriously. If you discover a security vulnerability, please follow these steps:

### 1. **Do Not** Open a Public Issue

Please **do not** report security vulnerabilities through public GitHub issues. Public disclosure could put the community at risk before a fix is available.

### 2. Report Privately

Instead, please report security vulnerabilities by:

- **Email**: Send details to the repository maintainer via GitHub's private vulnerability reporting feature
- **GitHub Security Advisories**: Use the "Report a vulnerability" button in the Security tab of this repository

### 3. What to Include

To help us address the issue quickly, please include:

- **Description**: A clear description of the vulnerability
- **Impact**: The potential impact and attack scenario
- **Reproduction Steps**: Detailed steps to reproduce the issue
- **Affected Components**: Which parts of the codebase are affected
- **Proof of Concept**: Code or screenshots demonstrating the vulnerability (if applicable)
- **Suggested Fix**: If you have ideas on how to fix it (optional)

### 4. Response Timeline

We are committed to responding promptly to security reports:

- **Initial Response**: Within 48 hours of receiving your report
- **Status Update**: We'll provide a detailed response within 7 days, including our assessment and planned next steps
- **Resolution**: We aim to release a fix within 30 days for critical vulnerabilities, depending on complexity

### 5. Disclosure Policy

We follow **coordinated disclosure**:

1. You report the vulnerability privately
2. We work on a fix and keep you updated on progress
3. Once a fix is ready and deployed, we'll coordinate with you on public disclosure timing
4. We'll credit you in the security advisory (unless you prefer to remain anonymous)

## Security Best Practices

When deploying DevWhisper:

### API Keys & Secrets

- **Never commit** API keys, tokens, or secrets to the repository
- Use `.env` files (which are in `.gitignore`) for all sensitive configuration
- Rotate `ADMIN_SECRET` regularly and use a strong, random value
- Generate admin secrets using: `python -c "import secrets; print(secrets.token_urlsafe(32))"`

### Network Security

- Always use HTTPS in production (configure your reverse proxy/load balancer)
- Use secure tunneling solutions (ngrok with authentication, or better alternatives)
- Restrict access to admin endpoints (`/admin/*`) through network-level controls
- Consider implementing additional authentication middleware for production deployments

### File Upload Security

- The `/upload` endpoint accepts ZIP files - ensure you trust the source
- File size and extraction limits are enforced (see `.env.example` for `MAX_UPLOAD_SIZE_MB`)
- Path traversal protection is built-in, but always run DevWhisper in a sandboxed environment

### API Security

- Set `ADMIN_SECRET` to protect admin endpoints
- Implement rate limiting at the reverse proxy level for production
- Monitor logs for suspicious activity
- Consider implementing IP allowlists for sensitive endpoints

### Dependency Security

- Regularly update dependencies: `pip install --upgrade -r requirements.txt`
- Monitor for security advisories on dependencies
- Run security scans: `pip install safety && safety check`

## Known Security Considerations

### LLM API Keys

DevWhisper requires API keys for:
- **Groq** (or OpenAI-compatible LLM provider)
- **Qdrant** (vector database)
- **Vapi** (voice platform, optional)

These keys provide access to external services and should be protected. If compromised:
1. Immediately rotate the affected keys in the service provider's dashboard
2. Update your `.env` file with the new keys
3. Restart the DevWhisper service

### Code Indexing Privacy

DevWhisper indexes your codebase and stores it in Qdrant. Be aware:
- Indexed code is stored in the configured Qdrant instance
- If using Qdrant Cloud, your code resides on their infrastructure
- For sensitive codebases, consider running a self-hosted Qdrant instance
- The `/upload` endpoint allows uploading codebases - restrict access appropriately

### Voice Data

If using Vapi integration:
- Voice data is processed by Vapi's infrastructure
- Review Vapi's privacy policy for data handling details
- Consider implications for sensitive discussions about proprietary code

## Security Updates

Security fixes will be released as:
- Patch commits on the `main` branch
- GitHub Security Advisories (for vulnerabilities)
- Updates in this SECURITY.md file

To stay informed:
- Watch this repository for releases
- Subscribe to security advisories through GitHub

## Acknowledgments

We appreciate the security research community's efforts to responsibly disclose vulnerabilities. Contributors who report valid security issues will be credited in our security advisories (unless they prefer anonymity).

Thank you for helping keep DevWhisper and its users safe!
