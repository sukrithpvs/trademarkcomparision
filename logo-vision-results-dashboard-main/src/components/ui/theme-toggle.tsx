
import * as React from "react"
import { Moon, Sun } from "lucide-react"
import { useTheme } from "../ThemeProvider"
import { Button } from "./button"
import { motion, AnimatePresence } from "framer-motion"

export function ThemeToggle() {
  const { theme, setTheme } = useTheme()

  const toggleTheme = () => {
    if (theme === 'light') {
      setTheme('dark')
    } else if (theme === 'dark') {
      setTheme('default')
    } else {
      setTheme('light')
    }
  }

  const getIcon = () => {
    switch (theme) {
      case 'light':
        return <Sun className="h-4 w-4" />
      case 'dark':
        return <Moon className="h-4 w-4" />
      default:
        return <div className="h-4 w-4 bg-gradient-nebula rounded-full animate-pulse" />
    }
  }

  const getThemeLabel = () => {
    switch (theme) {
      case 'light':
        return 'Light'
      case 'dark':
        return 'Dark'
      default:
        return 'Nebula'
    }
  }

  return (
    <motion.div
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
    >
      <Button
        variant="outline"
        size="sm"
        onClick={toggleTheme}
        className="glass-card hover:glow-primary transition-all duration-300 hover:scale-105 rounded-xl border-white/20 relative overflow-hidden group"
      >
        <div className="absolute inset-0 bg-gradient-nebula opacity-0 group-hover:opacity-10 transition-opacity duration-300"></div>
        <AnimatePresence mode="wait">
          <motion.div
            key={theme}
            initial={{ rotate: -90, opacity: 0 }}
            animate={{ rotate: 0, opacity: 1 }}
            exit={{ rotate: 90, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="flex items-center gap-2"
          >
            {getIcon()}
            <span className="text-sm font-medium">{getThemeLabel()}</span>
          </motion.div>
        </AnimatePresence>
      </Button>
    </motion.div>
  )
}
