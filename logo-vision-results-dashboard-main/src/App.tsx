import React, { useState, useEffect, useMemo } from 'react';
import { ThemeProvider } from './components/ThemeProvider';
import { Navbar } from './components/Navbar';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Search, 
  Settings, 
  Play, 
  BarChart3, 
  Shield, 
  ShieldAlert, 
  Eye, 
  FileText, 
  Download,
  Filter,
  RefreshCw,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Clock,
  Target,
  Zap,
  Sparkles,
  Brain,
  Cpu,
  Database,
  Activity,
  Award,
  Globe
} from "lucide-react";

const App = () => {
  const [activeTab, setActiveTab] = useState("analysis");
  const [analysisData, setAnalysisData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(0);
  const [formData, setFormData] = useState({
    referenceFolderPath: '',
    comparisonFolderPath: '',
    infringementThreshold: '70.0',
    batchSize: '512',
    maxImages: '2000',
    minimumSimilarityThreshold: '50.0'
  });

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const analyzeLogos = async () => {
    setLoading(true);
    setError(null);
    setProgress(0);
    
    // Simulate progress updates
    const progressInterval = setInterval(() => {
      setProgress(prev => Math.min(prev + Math.random() * 10, 90));
    }, 500);
    
    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          reference_folder_path: formData.referenceFolderPath,
          comparison_folder_path: formData.comparisonFolderPath,
          infringement_threshold: parseFloat(formData.infringementThreshold),
          batch_size: parseInt(formData.batchSize),
          max_images: parseInt(formData.maxImages)
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setAnalysisData(data);
      setProgress(100);
    } catch (err) {
      setError(err.message);
    } finally {
      clearInterval(progressInterval);
      setLoading(false);
    }
  };

  return (
    <ThemeProvider defaultTheme="default">
      <div className="min-h-screen bg-background particle-bg relative overflow-x-hidden">
        {/* Background Effects */}
        <div className="fixed inset-0 bg-gradient-radial from-primary/5 via-transparent to-accent/5 pointer-events-none"></div>
        <div className="fixed top-20 right-20 w-96 h-96 bg-primary/10 rounded-full blur-3xl animate-float pointer-events-none"></div>
        <div className="fixed bottom-20 left-20 w-80 h-80 bg-accent/10 rounded-full blur-3xl animate-float pointer-events-none" style={{animationDelay: '2s'}}></div>

        {/* NebulaFlow Navbar */}
        <Navbar />

        {/* Main Content */}
        <div className="relative">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            {/* Hero Section */}
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              className="relative mb-16 text-center"
            >
              <div className="absolute inset-0 bg-gradient-radial from-primary/20 via-transparent to-transparent animate-float"></div>
              <div className="relative">
                <motion.div 
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.6, delay: 0.2 }}
                  className="inline-flex items-center gap-2 mb-6 px-6 py-3 glass-card rounded-full border border-white/20"
                >
                  <Sparkles className="h-5 w-5 text-accent animate-pulse" />
                  <span className="text-sm font-medium bg-gradient-nebula bg-clip-text text-transparent">
                    Trademark
                  </span>
                </motion.div>
                
                <motion.h1 
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.8, delay: 0.4 }}
                  className="text-6xl lg:text-7xl font-heading font-bold mb-8 leading-tight"
                >
                  <span className="bg-gradient-nebula bg-clip-text text-transparent">
                    Logo Detection
                  </span>
                  <br />
                 
                </motion.h1>
                
                
                
                
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 1 }}
            >
              {/* Replace the Tabs structure with the new navigation */}
              <div className="space-y-8">
                {/* Navigation Buttons */}
                <div className="flex justify-center">
                  <div className="grid w-full grid-cols-2 lg:w-96 mx-auto glass-card border border-white/20 p-2 rounded-2xl">
                    <Button
                      variant={activeTab === "analysis" ? "default" : "ghost"}
                      onClick={() => setActiveTab("analysis")}
                      className={`
                        flex items-center gap-2 rounded-xl transition-all duration-300 hover:scale-105 tab-button-hover
                        ${activeTab === "analysis" 
                          ? "bg-gradient-nebula text-white shadow-lg glow-primary" 
                          : "hover:bg-white/10 hover:shadow-md"
                        }
                      `}
                    >
                      <Play className="h-4 w-4" />
                      Analysis Setup
                    </Button>
                    <Button
                      variant={activeTab === "results" ? "default" : "ghost"}
                      onClick={() => setActiveTab("results")}
                      className={`
                        flex items-center gap-2 rounded-xl transition-all duration-300 hover:scale-105 tab-button-hover
                        ${activeTab === "results" 
                          ? "bg-gradient-nebula text-white shadow-lg glow-primary" 
                          : "hover:bg-white/10 hover:shadow-md"
                        }
                      `}
                    >
                      <BarChart3 className="h-4 w-4" />
                      Results Dashboard
                    </Button>
                  </div>
                </div>

                {/* Content based on active tab */}
                <AnimatePresence mode="wait">
                  {activeTab === "analysis" && (
                    <motion.div
                      key="analysis"
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -20 }}
                      transition={{ duration: 0.3 }}
                    >
                      {/* Your existing Analysis Setup content */}
                      <Card className="glass-card border-0 shadow-2xl rounded-3xl overflow-hidden">
                        <CardHeader className="pb-8 bg-gradient-to-r from-primary/10 to-accent/10">
                          <CardTitle className="flex items-center gap-4 text-3xl">
                            <div className="p-3 bg-gradient-nebula rounded-2xl glow-primary">
                              <Settings className="h-7 w-7 text-white" />
                            </div>
                            <div>
                              <span className="bg-gradient-nebula bg-clip-text text-transparent">
                                Analysis Configuration
                              </span>
                              <p className="text-sm text-muted font-normal mt-2">
                                Configure your logo analysis parameters for optimal AI-powered results
                              </p>
                            </div>
                          </CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-10 p-8">
                          {/* Path Configuration */}
                          <div className="space-y-6">
                            <div className="flex items-center gap-3 mb-6">
                              <Database className="h-6 w-6 text-primary" />
                              <h3 className="text-xl font-heading font-semibold">Data Sources</h3>
                            </div>
                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                              <div className="space-y-4">
                                <Label htmlFor="reference" className="text-base font-semibold flex items-center gap-3">
                                  <div className="p-2 bg-primary/20 rounded-lg">
                                    <Target className="h-5 w-5 text-primary" />
                                  </div>
                                  Reference Folder Path
                                </Label>
                                <Input
                                  id="reference"
                                  name="referenceFolderPath"
                                  value={formData.referenceFolderPath}
                                  onChange={handleInputChange}
                                  placeholder="C:/path/to/reference/folder"
                                  className="h-14 glass-card border-0 focus:glow-primary transition-all duration-300 rounded-2xl text-lg"
                                />
                              </div>
                              <div className="space-y-4">
                                <Label htmlFor="comparison" className="text-base font-semibold flex items-center gap-3">
                                  <div className="p-2 bg-accent/20 rounded-lg">
                                    <Search className="h-5 w-5 text-accent" />
                                  </div>
                                  Comparison Folder Path
                                </Label>
                                <Input
                                  id="comparison"
                                  name="comparisonFolderPath"
                                  value={formData.comparisonFolderPath}
                                  onChange={handleInputChange}
                                  placeholder="C:/path/to/comparison/folder"
                                  className="h-14 glass-card border-0 focus:glow-accent transition-all duration-300 rounded-2xl text-lg"
                                />
                              </div>
                            </div>
                          </div>

                          <Separator className="opacity-20" />

                          {/* Advanced Settings */}
                          <div className="space-y-8">
                            <div className="flex items-center gap-4">
                              <div className="p-3 bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-2xl">
                                <Zap className="h-6 w-6 text-purple-400" />
                              </div>
                              <div>
                                <h3 className="text-xl font-heading font-semibold">Advanced AI Parameters</h3>
                                <p className="text-sm text-muted">Fine-tune neural network analysis parameters</p>
                              </div>
                            </div>
                            <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
                              <div className="space-y-4">
                                <Label className="text-sm font-medium text-muted flex items-center gap-2">
                                  <Shield className="h-4 w-4" />
                                  Infringement Threshold (%)
                                </Label>
                                <Input
                                  name="infringementThreshold"
                                  type="number"
                                  value={formData.infringementThreshold}
                                  onChange={handleInputChange}
                                  min="0"
                                  max="100"
                                  step="0.1"
                                  className="h-12 glass-card border-0 rounded-xl"
                                />
                              </div>
                              <div className="space-y-4">
                                <Label className="text-sm font-medium text-muted flex items-center gap-2">
                                  <TrendingUp className="h-4 w-4" />
                                  Min. Similarity (%)
                                </Label>
                                <Input
                                  name="minimumSimilarityThreshold"
                                  type="number"
                                  value={formData.minimumSimilarityThreshold}
                                  onChange={handleInputChange}
                                  min="0"
                                  max="100"
                                  step="0.1"
                                  className="h-12 glass-card border-0 rounded-xl"
                                />
                              </div>
                              <div className="space-y-4">
                                <Label className="text-sm font-medium text-muted flex items-center gap-2">
                                  <Activity className="h-4 w-4" />
                                  Batch Size
                                </Label>
                                <Input
                                  name="batchSize"
                                  type="number"
                                  value={formData.batchSize}
                                  onChange={handleInputChange}
                                  min="1"
                                  max="1024"
                                  className="h-12 glass-card border-0 rounded-xl"
                                />
                              </div>
                              <div className="space-y-4">
                                <Label className="text-sm font-medium text-muted flex items-center gap-2">
                                  <Database className="h-4 w-4" />
                                  Max Images
                                </Label>
                                <Input
                                  name="maxImages"
                                  type="number"
                                  value={formData.maxImages}
                                  onChange={handleInputChange}
                                  min="1"
                                  max="10000"
                                  className="h-12 glass-card border-0 rounded-xl"
                                />
                              </div>
                            </div>
                          </div>

                          {/* Analysis Button */}
                          <div className="flex flex-col items-center gap-8 pt-8">
                            <motion.div
                              whileHover={{ scale: 1.02 }}
                              whileTap={{ scale: 0.98 }}
                              className="w-full max-w-md"
                            >
                              <Button 
                                onClick={analyzeLogos} 
                                disabled={loading || !formData.referenceFolderPath || !formData.comparisonFolderPath}
                                size="lg"
                                className="w-full h-16 btn-gradient text-xl font-semibold hover:scale-105 transition-all duration-500 shadow-2xl rounded-2xl relative overflow-hidden group"
                              >
                                <div className="absolute inset-0 bg-gradient-to-r from-primary/20 to-accent/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                                {loading ? (
                                  <>
                                    <RefreshCw className="h-6 w-6 mr-3 animate-spin" />
                                    Analyzing with Neural Networks...
                                  </>
                                ) : (
                                  <>
                                    <Play className="h-6 w-6 mr-3" />
                                    Start Analysis
                                  </>
                                )}
                              </Button>
                            </motion.div>
                            
                            <AnimatePresence>
                              {loading && (
                                <motion.div 
                                  initial={{ opacity: 0, y: 20 }}
                                  animate={{ opacity: 1, y: 0 }}
                                  exit={{ opacity: 0, y: -20 }}
                                  className="w-full max-w-md space-y-6"
                                >
                                  <div className="relative">
                                    <Progress value={progress} className="h-4 bg-surface rounded-full overflow-hidden">
                                      <div 
                                        className="h-full bg-gradient-nebula transition-all duration-500 rounded-full glow-primary relative overflow-hidden"
                                        style={{ width: `${progress}%` }}
                                      >
                                        <div className="absolute inset-0 bg-white/20 animate-pulse"></div>
                                      </div>
                                    </Progress>
                                  </div>
                                  <div className="text-center space-y-2">
                                    <p className="text-lg font-medium">
                                      Processing with AI analysis... {progress.toFixed(0)}%
                                    </p>
                                    <p className="text-sm text-muted">
                                      Advanced computer vision models analyzing visual patterns and extracting semantic features
                                    </p>
                                    <div className="flex items-center justify-center gap-4 mt-4">
                                      <div className="flex items-center gap-2 text-xs text-muted">
                                        <Brain className="h-4 w-4 text-primary animate-pulse" />
                                        Vision AI
                                      </div>
                                      <div className="flex items-center gap-2 text-xs text-muted">
                                        <Cpu className="h-4 w-4 text-accent animate-pulse" />
                                        Text OCR
                                      </div>
                                      <div className="flex items-center gap-2 text-xs text-muted">
                                        <Award className="h-4 w-4 text-green-400 animate-pulse" />
                                        Legal Analysis
                                      </div>
                                    </div>
                                  </div>
                                </motion.div>
                              )}
                            </AnimatePresence>
                          </div>
                        </CardContent>
                      </Card>
                    </motion.div>
                  )}

                  {activeTab === "results" && (
                    <motion.div
                      key="results"
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -20 }}
                      transition={{ duration: 0.3 }}
                    >
                      {analysisData ? (
                        <AnalysisResults 
                          data={analysisData} 
                          minimumThreshold={parseFloat(formData.minimumSimilarityThreshold)}
                        />
                      ) : (
                        <Card className="glass-card border-0 shadow-2xl rounded-3xl">
                          {/* Keep your existing empty state content */}
                        </Card>
                      )}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </ThemeProvider>
  );
};

// Enhanced Stats Overview Component with NebulaFlow styling
const StatsOverview = ({ data, filteredData }) => {
  const stats = useMemo(() => {
    if (!data || !data.batch_results || !Array.isArray(data.batch_results)) {
      return {
        totalComparisons: 0,
        shownComparisons: 0,
        infringements: 0,
        infringementRate: 0,
        averageScore: 0,
        averageVisualSimilarity: 0,
        highRiskCount: 0,
        clearResults: 0
      };
    }

    if (!filteredData || !filteredData.batch_results || !Array.isArray(filteredData.batch_results)) {
      return {
        totalComparisons: 0,
        shownComparisons: 0,
        infringements: 0,
        infringementRate: 0,
        averageScore: 0,
        averageVisualSimilarity: 0,
        highRiskCount: 0,
        clearResults: 0
      };
    }

    const allResults = data.batch_results.flatMap(batch => batch.results || []);
    const filteredResults = filteredData.batch_results.flatMap(batch => batch.results || []);
    
    const totalComparisons = allResults.length;
    const shownComparisons = filteredResults.length;
    const infringements = filteredResults.filter(r => r.infringement_detected).length;
    const infringementRate = shownComparisons > 0 ? (infringements / shownComparisons) * 100 : 0;
    const averageScore = shownComparisons > 0 
      ? filteredResults.reduce((sum, r) => sum + (r.final_similarity || 0), 0) / shownComparisons 
      : 0;
    const averageVisualSimilarity = shownComparisons > 0 
      ? filteredResults.reduce((sum, r) => sum + (r.vit_similarity || 0), 0) / shownComparisons 
      : 0;
    const highRiskCount = filteredResults.filter(r => (r.final_similarity || 0) >= 70).length;

    return {
      totalComparisons,
      shownComparisons,
      infringements,
      infringementRate,
      averageScore,
      averageVisualSimilarity,
      highRiskCount,
      clearResults: shownComparisons - infringements
    };
  }, [data, filteredData]);

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-6 mb-8">
      <Card className="glass-card border-0 hover:scale-105 transition-all duration-300 hover:glow-primary">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-semibold">Total Processed</CardTitle>
          <BarChart3 className="h-5 w-5 text-primary" />
        </CardHeader>
        <CardContent>
          <div className="text-3xl font-heading font-bold text-primary">{stats.totalComparisons}</div>
          <p className="text-xs text-muted mt-1">Showing {stats.shownComparisons} above threshold</p>
        </CardContent>
      </Card>

      <Card className="glass-card border-0 hover:scale-105 transition-all duration-300 hover:shadow-red-500/20">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-semibold">Infringements</CardTitle>
          <ShieldAlert className="h-5 w-5 text-red-500" />
        </CardHeader>
        <CardContent>
          <div className="text-3xl font-heading font-bold text-red-500">{stats.infringements}</div>
          <div className="flex items-center gap-2 mt-1">
            <Badge variant="destructive" className="text-xs font-semibold">
              {stats.infringementRate.toFixed(1)}%
            </Badge>
            <span className="text-xs text-muted">of shown</span>
          </div>
        </CardContent>
      </Card>

      <Card className="glass-card border-0 hover:scale-105 transition-all duration-300 hover:glow-accent">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-semibold">Clear Results</CardTitle>
          <Shield className="h-5 w-5 text-accent" />
        </CardHeader>
        <CardContent>
          <div className="text-3xl font-heading font-bold text-accent">{stats.clearResults}</div>
          <div className="flex items-center gap-2 mt-1">
            <Badge className="text-xs font-semibold bg-accent/20 text-accent border-accent/30">
              {stats.shownComparisons > 0 ? ((stats.clearResults / stats.shownComparisons) * 100).toFixed(1) : 0}%
            </Badge>
            <span className="text-xs text-muted">safe</span>
          </div>
        </CardContent>
      </Card>

      <Card className="glass-card border-0 hover:scale-105 transition-all duration-300 hover:shadow-orange-500/20">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-semibold">Avg. Similarity</CardTitle>
          <TrendingUp className="h-5 w-5 text-orange-500" />
        </CardHeader>
        <CardContent>
          <div className="text-3xl font-heading font-bold text-orange-500">{stats.averageScore.toFixed(1)}%</div>
          <p className="text-xs text-muted mt-1">
            {stats.highRiskCount} high risk (≥70%)
          </p>
        </CardContent>
      </Card>

      <Card className="glass-card border-0 hover:scale-105 transition-all duration-300 hover:shadow-purple-500/20">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-semibold">Visual Analysis</CardTitle>
          <Eye className="h-5 w-5 text-purple-500" />
        </CardHeader>
        <CardContent>
          <div className="text-3xl font-heading font-bold text-purple-500">{stats.averageVisualSimilarity.toFixed(1)}%</div>
          <p className="text-xs text-muted mt-1">Average visual match</p>
        </CardContent>
      </Card>
    </div>
  );
};

// Enhanced Analysis Results Component
const AnalysisResults = ({ data, minimumThreshold }) => {
  const [selectedBatch, setSelectedBatch] = useState(0);
  const [sortBy, setSortBy] = useState('final_similarity');
  const [filterInfringement, setFilterInfringement] = useState(true);
  const [currentMinThreshold, setCurrentMinThreshold] = useState(minimumThreshold);
  const [showAllInfringements, setShowAllInfringements] = useState(false);

  const filteredData = useMemo(() => {
    if (!data || !data.batch_results || !Array.isArray(data.batch_results)) {
      return { batch_results: [] };
    }

    const filteredBatchResults = data.batch_results.map(batch => {
      if (!batch || !batch.results || !Array.isArray(batch.results)) {
        return {
          ...batch,
          results: [],
          infringement_count: 0
        };
      }

      const filteredResults = batch.results.filter(result => 
        result && (result.final_similarity || 0) >= currentMinThreshold
      );

      return {
        ...batch,
        results: filteredResults,
        infringement_count: filteredResults.filter(result => 
          result && result.infringement_detected
        ).length
      };
    });

    return {
      ...data,
      batch_results: filteredBatchResults
    };
  }, [data, currentMinThreshold]);

  const allInfringements = useMemo(() => {
    if (!filteredData.batch_results) return [];
    
    const infringements = [];
    filteredData.batch_results.forEach(batch => {
      if (batch.results) {
        batch.results.forEach(result => {
          if (result && result.infringement_detected) {
            infringements.push({
              ...result,
              reference_logo: batch.reference_logo
            });
          }
        });
      }
    });
    
    return infringements.sort((a, b) => (b.final_similarity || 0) - (a.final_similarity || 0));
  }, [filteredData]);

  useEffect(() => {
    if (!filteredData.batch_results || filteredData.batch_results.length === 0) {
      setSelectedBatch(0);
    } else if (selectedBatch >= filteredData.batch_results.length) {
      setSelectedBatch(0);
    }
  }, [filteredData.batch_results, selectedBatch]);

  if (!data || !filteredData.batch_results || filteredData.batch_results.length === 0) {
    return (
      <Card className="border-0 shadow-lg bg-white/70 backdrop-blur-sm">
        <CardContent className="flex flex-col items-center justify-center py-12">
          <AlertTriangle className="h-12 w-12 text-gray-400 mb-4" />
          <h3 className="text-lg font-semibold text-gray-900 mb-2">No Analysis Results</h3>
          <p className="text-gray-600 text-center">
            Please run an analysis first or check if your data contains valid results.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card className="border-0 shadow-lg bg-white/70 backdrop-blur-sm">
        <CardHeader>
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
            <div>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-blue-600" />
                Analysis Results
              </CardTitle>
              <div className="flex items-center gap-4 mt-2 text-sm text-gray-600">
                <div className="flex items-center gap-1">
                  <Clock className="h-4 w-4" />
                  Processing Time: {data.total_processing_time || 0}s
                </div>
                <div className="flex items-center gap-1">
                  <Target className="h-4 w-4" />
                  Min Threshold: {currentMinThreshold}%
                </div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant={showAllInfringements ? "default" : "outline"}
                onClick={() => setShowAllInfringements(!showAllInfringements)}
                className="flex items-center gap-2"
              >
                <ShieldAlert className="h-4 w-4" />
                {showAllInfringements ? 'Individual Results' : 'All Infringements'}
              </Button>
              {showAllInfringements && (
                <Badge variant="destructive" className="flex items-center gap-1">
                  {allInfringements.length} Total
                </Badge>
              )}
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Stats Overview */}
      <StatsOverview data={data} filteredData={filteredData} />

      {/* Results Content */}
      {showAllInfringements ? (
        <AllInfringementsView 
          infringements={allInfringements}
          data={data}
          sortBy={sortBy}
          setSortBy={setSortBy}
        />
      ) : (
        <IndividualResultsView 
          data={data}
          filteredData={filteredData}
          selectedBatch={selectedBatch}
          setSelectedBatch={setSelectedBatch}
          sortBy={sortBy}
          setSortBy={setSortBy}
          filterInfringement={filterInfringement}
          setFilterInfringement={setFilterInfringement}
          currentMinThreshold={currentMinThreshold}
          setCurrentMinThreshold={setCurrentMinThreshold}
        />
      )}
    </div>
  );
};

// Enhanced All Infringements View
const AllInfringementsView = ({ infringements, data, sortBy, setSortBy }) => {
  const sortedInfringements = useMemo(() => {
    return [...infringements].sort((a, b) => {
      if (sortBy === 'final_similarity') return (b.final_similarity || 0) - (a.final_similarity || 0);
      if (sortBy === 'text_similarity') return (b.text_similarity || 0) - (a.text_similarity || 0);
      if (sortBy === 'vit_similarity') return (b.vit_similarity || 0) - (a.vit_similarity || 0);
      return 0;
    });
  }, [infringements, sortBy]);

  return (
    <div className="space-y-6">
      {/* Controls */}
      <Card className="border-0 shadow-lg bg-white/70 backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Filter className="h-5 w-5 text-red-500" />
            All Infringements Overview
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
            <div className="flex items-center gap-2">
              <Label htmlFor="sort" className="text-sm font-medium">Sort by:</Label>
              <Select value={sortBy} onValueChange={setSortBy}>
                <SelectTrigger className="w-48">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="final_similarity">Final Similarity</SelectItem>
                  <SelectItem value="text_similarity">Text Similarity</SelectItem>
                  <SelectItem value="vit_similarity">Visual Similarity</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="text-sm text-gray-600">
              Showing <span className="font-semibold text-red-600">{sortedInfringements.length}</span> infringements across all reference images
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Results Grid */}
      {sortedInfringements.length === 0 ? (
        <Card className="border-0 shadow-lg bg-white/70 backdrop-blur-sm">
          <CardContent className="flex flex-col items-center justify-center py-12">
            <CheckCircle className="h-12 w-12 text-green-500 mb-4" />
            <h3 className="text-lg font-semibold text-gray-900 mb-2">No Infringements Detected</h3>
            <p className="text-gray-600 text-center">
              All logos have been analyzed and no potential infringements were found.
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
          {sortedInfringements.map((result, index) => (
            <EnhancedResultCard 
              key={`${result.reference_logo}-${result.logo_name}-${index}`}
              result={result}
              data={data}
              rank={index + 1}
              showReference={true}
              currentBatchIndex={0}
            />
          ))}
        </div>
      )}
    </div>
  );
};

// Enhanced Individual Results View
const IndividualResultsView = ({ 
  data, 
  filteredData, 
  selectedBatch, 
  setSelectedBatch, 
  sortBy, 
  setSortBy, 
  filterInfringement, 
  setFilterInfringement, 
  currentMinThreshold, 
  setCurrentMinThreshold
}) => {
  const currentBatch = filteredData.batch_results[selectedBatch];
  
  if (!currentBatch) {
    return (
      <Card className="border-0 shadow-lg bg-white/70 backdrop-blur-sm">
        <CardContent className="flex flex-col items-center justify-center py-12">
          <AlertTriangle className="h-12 w-12 text-gray-400 mb-4" />
          <h3 className="text-lg font-semibold text-gray-900 mb-2">No Results Found</h3>
          <p className="text-gray-600 text-center">
            Try adjusting your minimum similarity threshold.
          </p>
        </CardContent>
      </Card>
    );
  }

  const filteredResults = (currentBatch.results || []).filter(result => 
    filterInfringement ? (result && result.infringement_detected) : true
  );

  const sortedResults = [...filteredResults].sort((a, b) => {
    if (!a || !b) return 0;
    if (sortBy === 'final_similarity') return (b.final_similarity || 0) - (a.final_similarity || 0);
    if (sortBy === 'text_similarity') return (b.text_similarity || 0) - (a.text_similarity || 0);
    if (sortBy === 'vit_similarity') return (b.vit_similarity || 0) - (a.vit_similarity || 0);
    return 0;
  });

  return (
    <div className="space-y-6">
      {/* Controls */}
      <Card className="border-0 shadow-lg bg-white/70 backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Filter className="h-5 w-5 text-blue-600" />
            Filters & Controls
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Reference Selection */}
          <div className="space-y-2">
            <Label className="text-sm font-medium">Reference Image</Label>
            <Select value={selectedBatch.toString()} onValueChange={(value) => setSelectedBatch(parseInt(value))}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {filteredData.batch_results.map((batch, index) => (
                  <SelectItem key={index} value={index.toString()}>
                    {batch.reference_logo || `Batch ${index + 1}`} ({batch.infringement_count || 0} infringements, {(batch.results || []).length} results ≥{currentMinThreshold}%)
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="space-y-2">
              <Label className="text-sm font-medium">Sort by</Label>
              <Select value={sortBy} onValueChange={setSortBy}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="final_similarity">Final Similarity</SelectItem>
                  <SelectItem value="text_similarity">Text Similarity</SelectItem>
                  <SelectItem value="vit_similarity">Visual Similarity</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label className="text-sm font-medium">Minimum Similarity (%)</Label>
              <Input
                type="number"
                value={currentMinThreshold}
                onChange={(e) => setCurrentMinThreshold(parseFloat(e.target.value) || 0)}
                min="0"
                max="100"
                step="1"
              />
            </div>

            <div className="space-y-2">
              <Label className="text-sm font-medium">Filter Options</Label>
              <div className="flex items-center space-x-2 h-10">
                <Switch
                  id="infringement-filter"
                  checked={filterInfringement}
                  onCheckedChange={setFilterInfringement}
                />
                <Label htmlFor="infringement-filter" className="text-sm">
                  Infringements only
                </Label>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Results Summary */}
      <div className="text-sm text-gray-600 bg-white/50 backdrop-blur-sm rounded-lg p-4">
        Showing <span className="font-semibold text-gray-900">{sortedResults.length}</span> results with similarity ≥ {currentMinThreshold}% 
        for reference image: <span className="font-semibold text-blue-600">{currentBatch.reference_logo || 'Unknown'}</span>
      </div>

      {/* Results Grid */}
      {sortedResults.length === 0 ? (
        <Card className="border-0 shadow-lg bg-white/70 backdrop-blur-sm">
          <CardContent className="flex flex-col items-center justify-center py-12">
            <AlertTriangle className="h-12 w-12 text-gray-400 mb-4" />
            <h3 className="text-lg font-semibold text-gray-900 mb-2">No Results Found</h3>
            <p className="text-gray-600 text-center">
              Try lowering the minimum similarity threshold to see more results.
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
          {sortedResults.map((result, index) => (
            <EnhancedResultCard 
              key={`${result?.logo_name || 'unknown'}-${index}`}
              result={result}
              data={data}
              rank={index + 1}
              showReference={false}
              currentBatchIndex={selectedBatch}
            />
          ))}
        </div>
      )}
    </div>
  );
};

// Enhanced Result Card Component
const EnhancedResultCard = ({ result, data, rank, showReference = false, currentBatchIndex = 0 }) => {
  const [showDetails, setShowDetails] = useState(false);
  const [imageError, setImageError] = useState({ ref: false, comp: false });
  
  if (!result) {
    return (
      <Card className="border-0 shadow-lg bg-red-50/50 backdrop-blur-sm">
        <CardContent className="p-4">
          <p className="text-red-600">Error: Invalid result data</p>
        </CardContent>
      </Card>
    );
  }

  const constructImageUrl = (folderPath, imageName) => {
    if (!folderPath || !imageName) return '';
    
    let fullPath;
    if (imageName.includes('/') || imageName.includes('\\')) {
      fullPath = imageName;
    } else {
      fullPath = `${folderPath}/${imageName}`.replace(/\\/g, '/');
    }
    
    const encodedPath = encodeURIComponent(fullPath);
    return `http://localhost:8000/image/${encodedPath}`;
  };

  // Fixed reference image URL logic
  const referenceImageUrl = showReference 
    ? constructImageUrl(data.reference_folder_path, result.reference_logo)
    : constructImageUrl(data.reference_folder_path, data.batch_results[currentBatchIndex]?.reference_logo);
  
  const comparisonImageUrl = constructImageUrl(data.comparison_folder_path, result.logo_path);

  const getScoreColor = (score) => {
    if (score >= 70) return "text-red-600 bg-red-50";
    if (score >= 30) return "text-yellow-600 bg-yellow-50";
    return "text-green-600 bg-green-50";
  };

  return (
    <Card className={`border-0 shadow-lg transition-all duration-300 hover:shadow-xl hover:-translate-y-1 ${
      result.infringement_detected 
        ? 'bg-gradient-to-br from-red-50/80 to-orange-50/80 border-l-4 border-l-red-500' 
        : 'bg-white/80 border-l-4 border-l-green-500'
    } backdrop-blur-sm`}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-xs px-2 py-0.5">
              #{rank}
            </Badge>
            {result.infringement_detected ? (
              <ShieldAlert className="h-4 w-4 text-red-500" />
            ) : (
              <Shield className="h-4 w-4 text-green-500" />
            )}
            <span className="font-medium text-sm truncate">
              {result.logo_name}
            </span>
          </div>
          <Badge 
            variant={result.infringement_detected ? "destructive" : "default"}
            className="text-xs"
          >
            {result.infringement_detected ? "POTENTIAL" : "CLEAR"}
          </Badge>
        </div>
        
        {showReference && (
          <div className="text-xs text-gray-600 bg-gray-50 rounded px-2 py-1">
            Reference: {result.reference_logo}
          </div>
        )}
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Image Comparison */}
        <div className="relative rounded-lg overflow-hidden bg-gray-100 aspect-[16/9]">
          <div className="flex h-full">
            <div className="w-1/2 relative border-r border-gray-300">
              {imageError.ref ? (
                <div className="w-full h-full flex items-center justify-center bg-gray-50">
                  <div className="text-center">
                    <div className="text-2xl mb-1">📷</div>
                    <p className="text-xs text-gray-500">Reference</p>
                  </div>
                </div>
              ) : (
                <img 
                  src={referenceImageUrl}
                  alt="Reference Logo"
                  className="w-full h-full object-contain"
                  onError={() => setImageError(prev => ({ ...prev, ref: true }))}
                />
              )}
              <div className="absolute bottom-1 left-1 bg-black/70 text-white text-xs px-1.5 py-0.5 rounded">
                Reference
              </div>
            </div>
            <div className="w-1/2 relative">
              {imageError.comp ? (
                <div className="w-full h-full flex items-center justify-center bg-gray-50">
                  <div className="text-center">
                    <div className="text-2xl mb-1">📷</div>
                    <p className="text-xs text-gray-500">Comparison</p>
                  </div>
                </div>
              ) : (
                <img 
                  src={comparisonImageUrl}
                  alt="Comparison Logo"
                  className="w-full h-full object-contain"
                  onError={() => setImageError(prev => ({ ...prev, comp: true }))}
                />
              )}
              <div className="absolute bottom-1 right-1 bg-black/70 text-white text-xs px-1.5 py-0.5 rounded">
                Comparison
              </div>
            </div>
          </div>
          <div className="absolute top-1 left-1/2 transform -translate-x-1/2 bg-black/50 text-white text-xs px-2 py-0.5 rounded">
            vs
          </div>
        </div>

        {/* Final Score */}
        <div className="flex justify-between items-center">
          <span className="text-sm font-medium">Final Score</span>
          <Badge 
            variant={result.final_similarity >= 70 ? "destructive" : "default"}
            className="font-bold"
          >
            {(result.final_similarity || 0).toFixed(1)}%
          </Badge>
        </div>

        {/* Detailed Scores */}
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="flex justify-between">
            <span>Text:</span>
            <span className={`font-semibold px-2 py-0.5 rounded ${getScoreColor(result.text_similarity || 0)}`}>
              {(result.text_similarity || 0).toFixed(1)}%
            </span>
          </div>
          <div className="flex justify-between">
            <span>Visual:</span>
            <span className={`font-semibold px-2 py-0.5 rounded ${getScoreColor(result.vit_similarity || 0)}`}>
              {(result.vit_similarity || 0).toFixed(1)}%
            </span>
          </div>
        </div>

        {/* Text Comparison */}
        <div className="pt-2 border-t space-y-1">
          <div className="flex items-center gap-1 text-xs font-medium text-gray-600">
            <FileText className="h-3 w-3" />
            Extracted Text
          </div>
          <div className="text-xs space-y-1">
            <div className="flex justify-between">
              <span className="text-gray-500">Ref:</span>
              <span className="font-medium">"{result.text1 || 'N/A'}"</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Comp:</span>
              <span className="font-medium">"{result.text2 || 'N/A'}"</span>
            </div>
          </div>
        </div>

        {/* Toggle Details */}
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setShowDetails(!showDetails)}
          className="w-full flex items-center gap-2 text-xs h-8"
        >
          <Eye className="h-3 w-3" />
          {showDetails ? "Hide Details" : "Show Details"}
        </Button>

        {/* Additional Details */}
        {showDetails && (
          <div className="pt-2 border-t text-xs space-y-2">
            <div>
              <span className="font-medium text-gray-600">File Path:</span>
              <p className="text-gray-800 break-all">{result.logo_path || 'Unknown path'}</p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default App;
