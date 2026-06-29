import Navbar from './components/Navbar'
import Hero from './components/Hero'
import UploadAnalyze from './components/UploadAnalyze'
import HowItWorks from './components/HowItWorks'
import Interpret from './components/Interpret'
import Trust from './components/Trust'
import Footer from './components/Footer'
import './App.css'

function App() {
  return (
    <div className="app">
      <Navbar />
      <Hero />
      <UploadAnalyze />
      <HowItWorks />
      <Interpret />
      <Trust />
      <Footer />
    </div>
  )
}

export default App
