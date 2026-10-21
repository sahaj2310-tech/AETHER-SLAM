import React from 'react';

/**
 * Catches render-time errors in any child subtree so a single bad component
 * (most commonly a Three.js WebGL setup failure) doesn't blank the whole UI.
 * Renders a small in-place panel with the error message and preserves the
 * rest of the dashboard so you can still see telemetry, logs, etc.
 */
export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    // eslint-disable-next-line no-console
    console.error('[ErrorBoundary]', error, info?.componentStack);
  }

  reset = () => this.setState({ error: null });

  render() {
    if (this.state.error) {
      return (
        <div
          style={{
            width: '100%',
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 12,
            padding: 24,
            color: '#ff8888',
            background: 'rgba(40, 6, 6, 0.4)',
            border: '1px solid #ff2a2a',
            borderRadius: 8,
            fontFamily: 'var(--font-mono, monospace)',
            fontSize: 12,
            textAlign: 'center',
          }}
        >
          <div style={{ fontSize: 14, fontWeight: 700, color: '#ff2a2a' }}>
            {this.props.label || 'Component error'}
          </div>
          <div style={{ maxWidth: 600, whiteSpace: 'pre-wrap' }}>
            {String(this.state.error?.message || this.state.error)}
          </div>
          <button
            onClick={this.reset}
            style={{
              marginTop: 8,
              padding: '6px 14px',
              background: 'transparent',
              border: '1px solid #ff2a2a',
              color: '#ff8888',
              cursor: 'pointer',
              borderRadius: 4,
              fontFamily: 'inherit',
              letterSpacing: 1,
            }}
          >
            RETRY
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
