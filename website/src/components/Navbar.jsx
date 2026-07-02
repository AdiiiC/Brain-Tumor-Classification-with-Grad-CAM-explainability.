import { useState, useEffect } from 'react'
import './Navbar.css'

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false)
  const [dark, setDark] = useState(() => {
    return localStorage.getItem('theme') === 'dark'
  })

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 30)
    window.addEventListener('scroll', onScroll)
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light')
    localStorage.setItem('theme', dark ? 'dark' : 'light')
  }, [dark])

  return (
    <nav className={`navbar ${scrolled ? 'scrolled' : ''}`}>
      <div className="nav-container">
        <a href="#" className="nav-logo">
          <svg className="logo-icon" width="30" height="30" viewBox="0 0 32 32" fill="none" aria-hidden="true">
            <rect x="3.5" y="3.5" width="25" height="25" rx="3" stroke="var(--primary)" strokeWidth="1.6"/>
            <circle cx="16" cy="16" r="7" stroke="var(--primary)" strokeWidth="1.6"/>
            <path d="M16 2v6M16 24v6M2 16h6M24 16h6" stroke="var(--text-muted)" strokeWidth="1.4" strokeLinecap="round"/>
            <circle cx="18.5" cy="14" r="2.4" fill="var(--accent)"/>
          </svg>
          <span className="logo-text">Aperture<span className="logo-mark">/neuro</span></span>
          <span className="logo-badge">Clinical</span>
        </a>
        <div className="nav-links">
          <a href="#analyze" className="nav-link">Analyze Scan</a>
          <a href="#how-it-works" className="nav-link">How It Works</a>
          <a href="#interpret" className="nav-link">Reading Results</a>
          <a href="#results" className="nav-link">Results & Demo</a>
          <a href="#trust" className="nav-link">Safety & Accuracy</a>
          <button className="theme-toggle" onClick={() => setDark(!dark)} aria-label="Toggle dark mode">
            {dark ? (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/></svg>
            )}
          </button>
        </div>
      </div>
    </nav>
  )
}
