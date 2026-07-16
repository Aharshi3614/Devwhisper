const dot = document.getElementById('statusDot');
const label = document.getElementById('statusLabel');
const message = document.getElementById('statusMessage');
const lastUpdate = document.getElementById('lastUpdate');
const responseTime = document.getElementById('responseTime');

const STATUS_MAP = {
    healthy: {
        dotClass: 'healthy',
        label: '✅ Running',
        message: 'All services operational',
    },
    degraded: {
        dotClass: 'degraded',
        label: '⚠️ Degraded',
        message: 'Some services are slow or unavailable',
    },
    down: {
        dotClass: 'down',
        label: '❌ Service Down',
        message: 'Cannot connect to backend service',
    },
    unknown: {
        dotClass: '',
        label: '❓ Unknown',
        message: 'Unexpected status response',
    },
    error: {
        dotClass: 'down',
        label: '❌ Connection Failed',
        message: 'Request failed, check if backend is running',
    },
};

function updateStatus(status, msg) {
    const config = STATUS_MAP[status] || STATUS_MAP.unknown;

    dot.className = 'status-dot ' + config.dotClass;
    label.textContent = config.label;
    message.textContent = msg || config.message;
}

async function fetchHealth() {
    const startTime = performance.now();

    try {
        const res = await fetch('/health');

        const elapsed = Math.round(performance.now() - startTime);
        responseTime.textContent = elapsed + ' ms';

        if (!res.ok) {
            updateStatus('down', 'HTTP ' + res.status + ' ' + res.statusText);
            return;
        }

        const data = await res.json();

        const statusValue = data.status || data.state || 'unknown';

        if (statusValue === 'healthy' || statusValue === 'ok' || statusValue === 'up' || statusValue === 'DevWhisper is running') {
            updateStatus('healthy', data.message || 'All services operational');
        } else if (statusValue === 'degraded' || statusValue === 'warning') {
            updateStatus('degraded', data.message || 'Some services degraded');
        } else if (statusValue === 'down' || statusValue === 'error' || statusValue === 'unhealthy') {
            updateStatus('down', data.message || 'Service unavailable');
        } else {
            updateStatus('unknown', 'Status: ' + statusValue);
        }

    } catch (error) {
        responseTime.textContent = '-- ms';
        updateStatus('error', 'Connection failed: ' + error.message);
    }

    const now = new Date();
    lastUpdate.textContent = now.toLocaleTimeString('en-US', { hour12: false });
}

document.addEventListener('DOMContentLoaded', function() {
    fetchHealth();
    setInterval(fetchHealth, 3000);
});
