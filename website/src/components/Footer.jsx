import './Footer.css'

export default function Footer() {
  return (
    <footer className="site-footer">
      <div className="container">
        <div className="footer-content">
          <div className="footer-brand">
            <h3>BrainScan AI Clinical</h3>
            <p>AI-assisted brain tumor classification with explainability</p>
          </div>

          <div className="footer-links">
            <a href="#analyze">Analyze Scan</a>
            <a href="#how-it-works">How It Works</a>
            <a href="#interpret">Reading Results</a>
            <a href="#trust">Safety & Accuracy</a>
          </div>
        </div>

        <div className="footer-disclaimer">
          <p>
            <strong>Medical Disclaimer:</strong> This software is a research and clinical decision-support tool only.
            It is not a certified medical device and should not be used as the sole basis for clinical decisions.
            All AI-generated findings must be reviewed and confirmed by a qualified healthcare professional.
            The developers accept no liability for clinical decisions made based on this tool's output.
          </p>
        </div>

        <div className="footer-bottom">
          <p>&copy; {new Date().getFullYear()} BrainScan AI Clinical. For research and clinical decision support.</p>
        </div>
      </div>
    </footer>
  )
}
