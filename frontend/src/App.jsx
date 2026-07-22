import { BrowserRouter, Routes, Route, Link } from 'react-router-dom'
import HistoryPanel from './components/HistoryPanel.jsx'
import ThemeToggle from "./components/ThemeToggle"
import MicButton from './components/MicButton.jsx'
import "./App.css"

function Home() {
  return (
    <div className="home-content">
      <h1>DevWhisper</h1>
      <p>Voice-native developer experience agent</p>

      <MicButton />

      <p>
        <Link to="/history">View History</Link>
      </p>
    </div>
  )
}

function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <ThemeToggle />

        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/history" element={<HistoryPanel />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App