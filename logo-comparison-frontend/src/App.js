import React, { useState } from 'react';
import './App.css';

const App = () => {
  const [analysisData, setAnalysisData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [formData, setFormData] = useState({
    referenceFolderPath: '',
    comparisonFolderPath: '',
    infringementThreshold: 70.0,
    batchSize: 512,
    maxImages: 2000,
    minimumSimilarityThreshold: 50.0
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
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Logo Similarity Analysis</h1>
        <p>Compare logos and detect potential infringements with Visual and text analysis</p>
      </header>

      <div className="analysis-form">
        <div className="form-group">
          <label>Reference Folder Path:</label>
          <input
            type="text"
            name="referenceFolderPath"
            value={formData.referenceFolderPath}
            onChange={handleInputChange}
            placeholder="C:/path/to/reference/folder"
          />
        </div>

        <div className="form-group">
          <label>Comparison Folder Path:</label>
          <input
            type="text"
            name="comparisonFolderPath"
            value={formData.comparisonFolderPath}
            onChange={handleInputChange}
            placeholder="C:/path/to/comparison/folder"
          />
        </div>

        <div className="form-row">
          <div className="form-group">
            <label>Infringement Threshold (%):</label>
            <input
              type="number"
              name="infringementThreshold"
              value={formData.infringementThreshold}
              onChange={handleInputChange}
              min="0"
              max="100"
              step="0.1"
            />
          </div>

          <div className="form-group">
            <label>Minimum Similarity Threshold (%):</label>
            <input
              type="number"
              name="minimumSimilarityThreshold"
              value={formData.minimumSimilarityThreshold}
              onChange={handleInputChange}
              min="0"
              max="100"
              step="0.1"
              title="Only show results with final similarity above this threshold"
            />
          </div>

          <div className="form-group">
            <label>Batch Size:</label>
            <input
              type="number"
              name="batchSize"
              value={formData.batchSize}
              onChange={handleInputChange}
              min="1"
              max="1024"
            />
          </div>

          <div className="form-group">
            <label>Max Images:</label>
            <input
              type="number"
              name="maxImages"
              value={formData.maxImages}
              onChange={handleInputChange}
              min="1"
              max="10000"
            />
          </div>
        </div>

        <button 
          onClick={analyzeLogos} 
          disabled={loading || !formData.referenceFolderPath || !formData.comparisonFolderPath}
          className="analyze-button"
        >
          {loading ? 'Analyzing...' : 'Analyze Logos'}
        </button>
      </div>

      {error && (
        <div className="error-message">
          <h3>Error:</h3>
          <p>{error}</p>
        </div>
      )}

      {loading && (
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>Processing images with Visual analysis... This may take a while.</p>
        </div>
      )}

      {analysisData && (
        <AnalysisResults 
          data={analysisData} 
          minimumThreshold={formData.minimumSimilarityThreshold}
        />
      )}
    </div>
  );
};

// Updated Stats Overview Component
const StatsOverview = ({ data, filteredData }) => {
  const stats = React.useMemo(() => {
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
    <div className="stats-overview">
      <div className="stat-card">
        <div className="stat-header">
          <span className="stat-title">Total Processed</span>
          <div className="stat-icon"></div>
        </div>
        <div className="stat-value">{stats.totalComparisons}</div>
        <div className="stat-label">Showing {stats.shownComparisons} above threshold</div>
      </div>

      <div className="stat-card infringement">
        <div className="stat-header">
          <span className="stat-title">Infringements</span>
          <div className="stat-icon"></div>
        </div>
        <div className="stat-value">{stats.infringements}</div>
        <div className="stat-badge infringement-badge">
          {stats.infringementRate.toFixed(1)}% of shown
        </div>
      </div>

      <div className="stat-card clear">
        <div className="stat-header">
          <span className="stat-title">Clear Results</span>
          <div className="stat-icon"></div>
        </div>
        <div className="stat-value">{stats.clearResults}</div>
        <div className="stat-badge clear-badge">
          {stats.shownComparisons > 0 ? ((stats.clearResults / stats.shownComparisons) * 100).toFixed(1) : 0}% safe
        </div>
      </div>

      <div className="stat-card">
        <div className="stat-header">
          <span className="stat-title">Avg. Similarity</span>
          <div className="stat-icon"></div>
        </div>
        <div className="stat-value">{stats.averageScore.toFixed(1)}%</div>
        <div className="stat-label">{stats.highRiskCount} high risk (≥70%)</div>
      </div>

      <div className="stat-card visual">
        <div className="stat-header">
          <span className="stat-title">Visual Analysis</span>
          <div className="stat-icon"></div>
        </div>
        <div className="stat-value">{stats.averageVisualSimilarity.toFixed(1)}%</div>
        <div className="stat-label">Average Visual match</div>
      </div>
    </div>
  );
};

const AnalysisResults = ({ data, minimumThreshold }) => {
  const [selectedBatch, setSelectedBatch] = useState(0);
  const [sortBy, setSortBy] = useState('final_similarity');
  const [filterInfringement, setFilterInfringement] = useState(true);
  const [currentMinThreshold, setCurrentMinThreshold] = useState(minimumThreshold);
  const [showAllInfringements, setShowAllInfringements] = useState(false); // NEW STATE

  const filteredData = React.useMemo(() => {
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

  // NEW: Get all infringements across all batches
  const allInfringements = React.useMemo(() => {
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

  React.useEffect(() => {
    if (!filteredData.batch_results || filteredData.batch_results.length === 0) {
      setSelectedBatch(0);
    } else if (selectedBatch >= filteredData.batch_results.length) {
      setSelectedBatch(0);
    }
  }, [filteredData.batch_results, selectedBatch]);

  if (!data || !filteredData.batch_results || filteredData.batch_results.length === 0) {
    return (
      <div className="analysis-results">
        <div className="no-results-message">
          <p>⚠️ No analysis results available.</p>
          <p>Please run an analysis first or check if your data contains valid results.</p>
        </div>
      </div>
    );
  }

  const currentBatch = filteredData.batch_results[selectedBatch];
  
  if (!currentBatch) {
    return (
      <div className="analysis-results">
        <div className="no-results-message">
          <p>⚠️ No batch results found for the selected index.</p>
          <p>Try adjusting your minimum similarity threshold.</p>
        </div>
      </div>
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

  const referenceImageUrl = constructImageUrl(data.reference_folder_path, currentBatch.reference_logo);

  return (
    <div className="analysis-results">
      <div className="results-header">
        <div className="header-top">
          <h2>Analysis Results</h2>
          {/* NEW: Show All Infringements Button */}
          <div className="show-all-infringements-section">
            <button 
              className={`show-all-btn ${showAllInfringements ? 'active' : ''}`}
              onClick={() => setShowAllInfringements(!showAllInfringements)}
            >
              {showAllInfringements ? '📋 Show Individual Results' : '🚨 Show All Infringements'}
            </button>
            {showAllInfringements && (
              <span className="infringement-count-badge">
                {allInfringements.length} Total Infringements
              </span>
            )}
          </div>
        </div>
        
        <StatsOverview data={data} filteredData={filteredData} />
        
        <div className="analysis-method">
          <p><strong>Analysis Method:</strong> Visual + Text Analysis</p>
          <p><strong>Processing Time:</strong> {data.total_processing_time || 0}s</p>
          <p><strong>Minimum Similarity Threshold:</strong> {currentMinThreshold}%</p>
        </div>
      </div>

      {/* NEW: Conditional rendering based on showAllInfringements */}
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
          currentBatch={currentBatch}
          sortBy={sortBy}
          setSortBy={setSortBy}
          filterInfringement={filterInfringement}
          setFilterInfringement={setFilterInfringement}
          currentMinThreshold={currentMinThreshold}
          setCurrentMinThreshold={setCurrentMinThreshold}
          sortedResults={sortedResults}
          referenceImageUrl={referenceImageUrl}
        />
      )}
    </div>
  );
};

// NEW: All Infringements View Component
const AllInfringementsView = ({ infringements, data, sortBy, setSortBy }) => {
  const sortedInfringements = React.useMemo(() => {
    return [...infringements].sort((a, b) => {
      if (sortBy === 'final_similarity') return (b.final_similarity || 0) - (a.final_similarity || 0);
      if (sortBy === 'text_similarity') return (b.text_similarity || 0) - (a.text_similarity || 0);
      if (sortBy === 'vit_similarity') return (b.vit_similarity || 0) - (a.vit_similarity || 0);
      return 0;
    });
  }, [infringements, sortBy]);

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

  return (
    <div className="all-infringements-view">
      <div className="all-infringements-header">
        <h3>🚨 All Infringements Across All Reference Images</h3>
        <div className="filters">
          <div className="filter-group">
            <label>Sort by:</label>
            <select value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
              <option value="final_similarity">Final Similarity</option>
              <option value="text_similarity">Text Similarity</option>
              <option value="vit_similarity">Visual Similarity</option>
            </select>
          </div>
        </div>
      </div>

      <div className="results-summary">
        <p>
          Showing <strong>{sortedInfringements.length}</strong> infringements across all reference images
        </p>
      </div>

      {sortedInfringements.length === 0 ? (
        <div className="no-results-message">
          <p>✅ No infringements detected across all reference images.</p>
        </div>
      ) : (
        <div className="comparison-grid">
          {sortedInfringements.map((result, index) => {
            const referenceImageUrl = constructImageUrl(data.reference_folder_path, result.reference_logo);
            
            return (
              <AllInfringementsResultCard 
                key={`${result.reference_logo}-${result.logo_name}-${index}`}
                result={result}
                comparisonFolderPath={data.comparison_folder_path}
                referenceImageUrl={referenceImageUrl}
                rank={index + 1}
              />
            );
          })}
        </div>
      )}
    </div>
  );
};

// NEW: All Infringements Result Card Component
const AllInfringementsResultCard = ({ result, comparisonFolderPath, referenceImageUrl, rank }) => {
  const [showDetails, setShowDetails] = useState(false);
  const [imageError, setImageError] = useState({ ref: false, comp: false });
  
  if (!result) {
    return (
      <div className="comparison-card-new error">
        <p>Error: Invalid result data</p>
      </div>
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

  const comparisonImageUrl = constructImageUrl(comparisonFolderPath, result.logo_path);

  const handleImageError = (type) => {
    setImageError(prev => ({ ...prev, [type]: true }));
  };

  return (
    <div className="comparison-card-new infringement">
      {/* Header with Rank and Reference Info */}
      <div className="card-header-new">
        <div className="card-title-section">
          <div className="rank-badge">#{rank}</div>
          <div className="status-icon">⚠️</div>
          <span className="card-title">{result.logo_name || 'Unknown'}</span>
        </div>
        <div className="infringement-badge active">
          POTENTIAL SIMILARITY
        </div>
      </div>

      {/* Reference Logo Info */}
      <div className="reference-info">
        <span className="reference-label">Reference:</span>
        <span className="reference-name">{result.reference_logo}</span>
      </div>

      {/* Side-by-side Image Comparison */}
      <div className="image-comparison-section">
        <div className="comparison-overlay">
          <span className="vs-text">vs {result.logo_name || 'Unknown'}</span>
        </div>
        
        <div className="images-container">
          <div className="image-half reference-half">
            {imageError.ref ? (
              <div className="image-placeholder">
                <span>📷</span>
                <p>Reference Image</p>
              </div>
            ) : (
              <img 
                src={referenceImageUrl}
                alt="Reference Logo"
                className="comparison-img"
                onError={() => handleImageError('ref')}
              />
            )}
            <div className="image-label">Reference: {result.reference_logo}</div>
          </div>

          <div className="divider-line"></div>

          <div className="image-half comparison-half">
            {imageError.comp ? (
              <div className="image-placeholder">
                <span>📷</span>
                <p>Comparison Image</p>
              </div>
            ) : (
              <img 
                src={comparisonImageUrl}
                alt="Comparison Logo"
                className="comparison-img"
                onError={() => handleImageError('comp')}
              />
            )}
            <div className="image-label">Comparison</div>
          </div>
        </div>
      </div>

      {/* Final Score */}
      <div className="final-score-section">
        <span className="score-label">Final Score</span>
        <div className="score-badge high-score">
          {(result.final_similarity || 0).toFixed(1)}%
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="metrics-grid">
        <div className="metric-item">
          <span className="metric-label">Text:</span>
          <span className="metric-value">{(result.text_similarity || 0).toFixed(1)}%</span>
        </div>
        <div className="metric-item">
          <span className="metric-label">Visual:</span>
          <span className="metric-value">{(result.vit_similarity || 0).toFixed(1)}%</span>
        </div>
      </div>

      {/* Word Mark Section */}
      <div className="word-mark-section">
        <div className="section-header">
          <span className="text-icon">📝</span>
          <span>Word Mark</span>
        </div>
        <div className="text-comparison">
          <div className="text-row">
            <span className="text-label">Reference:</span>
            <div className="text-content">{result.text1 || 'No text detected'}</div>
          </div>
          <div className="text-row">
            <span className="text-label">Comparison:</span>
            <div className="text-content">{result.text2 || 'No text detected'}</div>
          </div>
        </div>
      </div>

      {/* Toggle Details Button */}
      <button 
        className="details-toggle-btn"
        onClick={() => setShowDetails(!showDetails)}
      >
        <span className="eye-icon">👁️</span>
        <span>{showDetails ? 'Hide Details' : 'Show Details'}</span>
      </button>

      {/* Additional Details */}
      {showDetails && (
        <div className="additional-details">
          <div className="detail-item">
            <strong>File Path:</strong>
            <p>{result.logo_path || 'Unknown path'}</p>
          </div>
          <div className="detail-item">
            <strong>Reference Image:</strong>
            <p>{result.reference_logo || 'Unknown reference'}</p>
          </div>
        </div>
      )}
    </div>
  );
};

// Individual Results View Component (existing functionality)
const IndividualResultsView = ({ 
  data, 
  filteredData, 
  selectedBatch, 
  setSelectedBatch, 
  currentBatch, 
  sortBy, 
  setSortBy, 
  filterInfringement, 
  setFilterInfringement, 
  currentMinThreshold, 
  setCurrentMinThreshold, 
  sortedResults, 
  referenceImageUrl 
}) => {
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

  return (
    <>
      <div className="reference-selection">
        <div className="batch-selector-with-preview">
          <div className="selector-section">
            <label>Select Reference Image:</label>
            <select 
              value={selectedBatch} 
              onChange={(e) => setSelectedBatch(parseInt(e.target.value))}
            >
              {filteredData.batch_results.map((batch, index) => (
                <option key={index} value={index}>
                  {batch.reference_logo || `Batch ${index + 1}`} ({batch.infringement_count || 0} infringements, {(batch.results || []).length} results ≥{currentMinThreshold}%)
                </option>
              ))}
            </select>
          </div>
          
          <div className="reference-preview">
            <h4>Reference Image Preview:</h4>
            <div className="reference-image-container">
              <img 
                src={referenceImageUrl} 
                alt={`Reference: ${currentBatch.reference_logo || 'Unknown'}`}
                className="reference-preview-image"
                onError={(e) => {
                  e.target.style.display = 'none';
                  e.target.nextSibling.style.display = 'block';
                }}
              />
              <div className="image-error-fallback" style={{display: 'none'}}>
                <p>Error loading reference image</p>
                <small>{currentBatch.reference_logo || 'Unknown'}</small>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="filters">
        <div className="filter-group">
          <label>Sort by:</label>
          <select value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
            <option value="final_similarity">Final Similarity</option>
            <option value="text_similarity">Text Similarity</option>
            <option value="vit_similarity">Visual Similarity</option>
          </select>
        </div>
        
        <div className="filter-group">
          <label>
            <input
              type="checkbox"
              checked={filterInfringement}
              onChange={(e) => setFilterInfringement(e.target.checked)}
            />
            Show only infringements
          </label>
        </div>

        <div className="filter-group">
          <label>Min Similarity:</label>
          <input
            type="number"
            value={currentMinThreshold}
            onChange={(e) => setCurrentMinThreshold(parseFloat(e.target.value) || 0)}
            min="0"
            max="100"
            step="1"
            style={{ width: '80px' }}
          />
          <span>%</span>
        </div>
      </div>

      <div className="results-summary">
        <p>
          Showing <strong>{sortedResults.length}</strong> results with similarity ≥ {currentMinThreshold}% 
          for reference image: <strong>{currentBatch.reference_logo || 'Unknown'}</strong>
        </p>
        {(currentBatch.results || []).length === 0 && (
          <div className="no-results-message">
            <p>⚠️ No results found above {currentMinThreshold}% similarity threshold.</p>
            <p>Try lowering the minimum similarity threshold to see more results.</p>
          </div>
        )}
      </div>

      <div className="comparison-grid">
        {sortedResults.map((result, index) => (
          <ResultCard 
            key={`${result?.logo_name || 'unknown'}-${index}`}
            result={result}
            comparisonFolderPath={data.comparison_folder_path}
            referenceImageUrl={referenceImageUrl}
            rank={index + 1}
          />
        ))}
      </div>
    </>
  );
};

// Keep the existing ResultCard component unchanged
const ResultCard = ({ result, comparisonFolderPath, referenceImageUrl, rank }) => {
  const [showDetails, setShowDetails] = useState(false);
  const [imageError, setImageError] = useState({ ref: false, comp: false });
  
  if (!result) {
    return (
      <div className="comparison-card-new error">
        <p>Error: Invalid result data</p>
      </div>
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

  const comparisonImageUrl = constructImageUrl(comparisonFolderPath, result.logo_path);

  const handleImageError = (type) => {
    setImageError(prev => ({ ...prev, [type]: true }));
  };

  return (
    <div className={`comparison-card-new ${result.infringement_detected ? 'infringement' : ''}`}>
      <div className="card-header-new">
        <div className="card-title-section">
          <div className="rank-badge">#{rank}</div>
          <div className="status-icon">
            {result.infringement_detected ? '⚠️' : '✅'}
          </div>
          <span className="card-title">{result.logo_name || 'Unknown'}</span>
        </div>
        <div className={`infringement-badge ${result.infringement_detected ? 'active' : ''}`}>
          {result.infringement_detected ? 'POTENTIAL SIMILARITY' : 'CLEAR'}
        </div>
      </div>

      <div className="image-comparison-section">
        <div className="comparison-overlay">
          <span className="vs-text">vs {result.logo_name || 'Unknown'}</span>
        </div>
        
        <div className="images-container">
          <div className="image-half reference-half">
            {imageError.ref ? (
              <div className="image-placeholder">
                <span>📷</span>
                <p>Reference Image</p>
              </div>
            ) : (
              <img 
                src={referenceImageUrl}
                alt="Reference Logo"
                className="comparison-img"
                onError={() => handleImageError('ref')}
              />
            )}
            <div className="image-label">Reference</div>
          </div>

          <div className="divider-line"></div>

          <div className="image-half comparison-half">
            {imageError.comp ? (
              <div className="image-placeholder">
                <span>📷</span>
                <p>Comparison Image</p>
              </div>
            ) : (
              <img 
                src={comparisonImageUrl}
                alt="Comparison Logo"
                className="comparison-img"
                onError={() => handleImageError('comp')}
              />
            )}
            <div className="image-label">Comparison</div>
          </div>
        </div>
      </div>

      <div className="final-score-section">
        <span className="score-label">Final Score</span>
        <div className={`score-badge ${(result.final_similarity || 0) >= 70 ? 'high-score' : 'low-score'}`}>
          {(result.final_similarity || 0).toFixed(1)}%
        </div>
      </div>

      <div className="metrics-grid">
        <div className="metric-item">
          <span className="metric-label">Text:</span>
          <span className="metric-value">{(result.text_similarity || 0).toFixed(1)}%</span>
        </div>
        <div className="metric-item">
          <span className="metric-label">Visual:</span>
          <span className="metric-value">{(result.vit_similarity || 0).toFixed(1)}%</span>
        </div>
      </div>

      <div className="word-mark-section">
        <div className="section-header">
          <span className="text-icon">📝</span>
          <span>Word Mark</span>
        </div>
        <div className="text-comparison">
          <div className="text-row">
            <span className="text-label">Reference:</span>
            <div className="text-content">{result.text1 || 'No text detected'}</div>
          </div>
          <div className="text-row">
            <span className="text-label">Comparison:</span>
            <div className="text-content">{result.text2 || 'No text detected'}</div>
          </div>
        </div>
      </div>

      <button 
        className="details-toggle-btn"
        onClick={() => setShowDetails(!showDetails)}
      >
        <span className="eye-icon">👁️</span>
        <span>{showDetails ? 'Hide Details' : 'Show Details'}</span>
      </button>

      {showDetails && (
        <div className="additional-details">
          <div className="detail-item">
            <strong>File Path:</strong>
            <p>{result.logo_path || 'Unknown path'}</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
