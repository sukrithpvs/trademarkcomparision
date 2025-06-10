
import React, { useState, useEffect } from 'react'
import { ThemeToggle } from './ui/theme-toggle'
import { BarChart3, Zap, Settings, Menu, X } from 'lucide-react'
import { Button } from './ui/button'

export function Navbar() {
  const [isScrolled, setIsScrolled] = useState(false)
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20)
    }

    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  return (
    <>
      {/* Floating Navbar */}
      <nav className={`
        fixed top-4 left-1/2 transform -translate-x-1/2 z-50 transition-all duration-500 ease-out
        ${isScrolled 
          ? 'glass-card backdrop-blur-xl border border-white/10 shadow-2xl scale-95' 
          : 'bg-transparent scale-100'
        }
        rounded-2xl px-6 py-3 max-w-6xl w-[95%] mx-auto
      `}>
        <div className="flex items-center justify-between">
          {/* Logo Section */}
          <div className="flex items-center gap-3">
            <div className="relative group">
              <div className="p-2 bg-gradient-nebula rounded-xl glow-primary animate-glow-pulse group-hover:scale-110 transition-transform duration-300">
                <BarChart3 className="h-6 w-6 text-white" />
              </div>
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-accent rounded-full animate-pulse"></div>
            </div>
            <div className="hidden sm:block">
              <h1 className="text-xl font-heading font-bold bg-gradient-nebula bg-clip-text text-transparent">
                Logo Detection
              </h1>
              <p className="text-xs text-muted">AI Logo Analysis</p>
            </div>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-8">
            <div className="flex items-center gap-6">
              <Button variant="ghost" size="sm" className="hover:glow-primary transition-all duration-300 hover:scale-105 rounded-xl">
                <BarChart3 className="h-4 w-4 mr-2" />
                Analytics
              </Button>
              <Button variant="ghost" size="sm" className="hover:glow-accent transition-all duration-300 hover:scale-105 rounded-xl">
                <Settings className="h-4 w-4 mr-2" />
                Settings
              </Button>
            </div>
            
            <div className="w-px h-6 bg-border/50"></div>
            
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 text-xs text-muted bg-surface/20 px-3 py-1.5 rounded-full border border-white/10">
                <Zap className="h-3 w-3 text-accent animate-pulse" />
                AI Enhanced
              </div>
              <ThemeToggle />
            </div>
          </div>

          {/* Mobile Menu Button */}
          <div className="md:hidden flex items-center gap-2">
            <ThemeToggle />
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="hover:glow-primary transition-all duration-300 rounded-xl"
            >
              {isMobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </Button>
          </div>
        </div>

        {/* Mobile Menu */}
        {isMobileMenuOpen && (
          <div className="md:hidden mt-4 pt-4 border-t border-white/10 animate-scale-in">
            <div className="flex flex-col gap-3">
              <Button variant="ghost" size="sm" className="justify-start hover:glow-primary transition-all duration-300 rounded-xl">
                <BarChart3 className="h-4 w-4 mr-2" />
                Analytics
              </Button>
              <Button variant="ghost" size="sm" className="justify-start hover:glow-accent transition-all duration-300 rounded-xl">
                <Settings className="h-4 w-4 mr-2" />
                Settings
              </Button>
              <div className="flex items-center gap-2 text-xs text-muted bg-surface/20 px-3 py-2 rounded-xl border border-white/10 mt-2">
                <Zap className="h-3 w-3 text-accent animate-pulse" />
                AI Enhanced Platform
              </div>
            </div>
          </div>
        )}
      </nav>

      {/* Spacer to prevent content overlap */}
      <div className="h-20"></div>
    </>
  )
}
