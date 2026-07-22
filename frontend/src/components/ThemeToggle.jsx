import { useEffect, useState } from "react";
import "./ThemeToggle.css";

export default function ThemeToggle() {
  const [theme, setTheme] = useState("light");

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  return (
    <label className="theme">
      <span className="theme__toggle-wrap">
        <input
          type="checkbox"
          className="theme__toggle"
          checked={theme === "dark"}
          onChange={() =>
            setTheme((prev) => (prev === "light" ? "dark" : "light"))
          }

          aria-label = "Toggle light and dark theme"
        />

        <span className="theme__fill"></span>

        <span className="theme__icon">
          <span className="theme__icon-part"></span>
          <span className="theme__icon-part"></span>
          <span className="theme__icon-part"></span>
          <span className="theme__icon-part"></span>
          <span className="theme__icon-part"></span>
          <span className="theme__icon-part"></span>
          <span className="theme__icon-part"></span>
          <span className="theme__icon-part"></span>
          <span className="theme__icon-part"></span>
        </span>
      </span>
    </label>
  );
}