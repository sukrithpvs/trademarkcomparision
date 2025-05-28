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
    minimumSimilarityThreshold: 50.0  // New field for filtering
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
        <p>Compare logos and detect potential infringements with strict color analysis</p>
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
          <p>Processing images with strict color analysis... This may take a while.</p>
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

// Updated Stats Overview Component with filtered data
const StatsOverview = ({ data, filteredData }) => {
  const stats = React.useMemo(() => {
    const allResults = data.batch_results.flatMap(batch => batch.results);
    const filteredResults = filteredData.batch_results.flatMap(batch => batch.results);
    
    const totalComparisons = allResults.length;
    const shownComparisons = filteredResults.length;
    const infringements = filteredResults.filter(r => r.infringement_detected).length;
    const infringementRate = shownComparisons > 0 ? (infringements / shownComparisons) * 100 : 0;
    const averageScore = shownComparisons > 0 
      ? filteredResults.reduce((sum, r) => sum + r.final_similarity, 0) / shownComparisons 
      : 0;
    const averageColorSimilarity = shownComparisons > 0 
      ? filteredResults.reduce((sum, r) => sum + r.color_similarity, 0) / shownComparisons 
      : 0;
    const highRiskCount = filteredResults.filter(r => r.final_similarity >= 70).length;

    return {
      totalComparisons,
      shownComparisons,
      infringements,
      infringementRate,
      averageScore,
      averageColorSimilarity,
      highRiskCount,
      clearResults: shownComparisons - infringements
    };
  }, [data, filteredData]);

  return (
    <div className="stats-overview">
      <div className="stat-card">
        <div className="stat-header">
          <span className="stat-title">Total Processed</span>
          <div className="stat-icon">📊</div>
        </div>
        <div className="stat-value">{stats.totalComparisons}</div>
        <div className="stat-label">Showing {stats.shownComparisons} above threshold</div>
      </div>

      <div className="stat-card infringement">
        <div className="stat-header">
          <span className="stat-title">Infringements</span>
          <div className="stat-icon">🚨</div>
        </div>
        <div className="stat-value">{stats.infringements}</div>
        <div className="stat-badge infringement-badge">
          {stats.infringementRate.toFixed(1)}% of shown
        </div>
      </div>

      <div className="stat-card clear">
        <div className="stat-header">
          <span className="stat-title">Clear Results</span>
          <div className="stat-icon">🛡️</div>
        </div>
        <div className="stat-value">{stats.clearResults}</div>
        <div className="stat-badge clear-badge">
          {((stats.clearResults / stats.shownComparisons) * 100).toFixed(1)}% safe
        </div>
      </div>

      <div className="stat-card">
        <div className="stat-header">
          <span className="stat-title">Avg. Similarity</span>
          <div className="stat-icon">📈</div>
        </div>
        <div className="stat-value">{stats.averageScore.toFixed(1)}%</div>
        <div className="stat-label">{stats.highRiskCount} high risk (≥70%)</div>
      </div>

      <div className="stat-card color">
        <div className="stat-header">
          <span className="stat-title">Color Analysis</span>
          <div className="stat-icon">🎨</div>
        </div>
        <div className="stat-value">{stats.averageColorSimilarity.toFixed(1)}%</div>
        <div className="stat-label">Average color match</div>
      </div>
    </div>
  );
};

const AnalysisResults = ({ data, minimumThreshold }) => {
  const [selectedBatch, setSelectedBatch] = useState(0);
  const [sortBy, setSortBy] = useState('final_similarity');
  const [filterInfringement, setFilterInfringement] = useState(false);
  const [currentMinThreshold, setCurrentMinThreshold] = useState(minimumThreshold);

  // Filter data based on minimum similarity threshold
  const filteredData = React.useMemo(() => {
    const filteredBatchResults = data.batch_results.map(batch => ({
      ...batch,
      results: batch.results.filter(result => 
        result.final_similarity >= currentMinThreshold
      ),
      // Update infringement count for filtered results
      infringement_count: batch.results.filter(result => 
        result.final_similarity >= currentMinThreshold && result.infringement_detected
      ).length
    }));

    return {
      ...data,
      batch_results: filteredBatchResults
    };
  }, [data, currentMinThreshold]);

  const currentBatch = filteredData.batch_results[selectedBatch];
  
  const filteredResults = currentBatch?.results.filter(result => 
    filterInfringement ? result.infringement_detected : true
  ) || [];

  const sortedResults = [...filteredResults].sort((a, b) => {
    if (sortBy === 'final_similarity') return b.final_similarity - a.final_similarity;
    if (sortBy === 'text_similarity') return b.text_similarity - a.text_similarity;
    if (sortBy === 'image_score') return b.image_score - a.image_score;
    if (sortBy === 'color_similarity') return b.color_similarity - a.color_similarity;
    if (sortBy === 'vit_similarity') return b.vit_similarity - a.vit_similarity;
    return 0;
  });

  // Helper function to construct proper image URLs
  const constructImageUrl = (folderPath, imageName) => {
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
        <h2>Analysis Results</h2>
        
        {/* Stats Overview with filtered data */}
        <StatsOverview data={data} filteredData={filteredData} />
        
        <div className="analysis-method">
          <p><strong>Analysis Method:</strong> {data.summary.analysis_method}</p>
          <p><strong>Processing Time:</strong> {data.total_processing_time}s</p>
          <p><strong>Minimum Similarity Threshold:</strong> {currentMinThreshold}%</p>
        </div>
      </div>

      {/* Reference Image Selection with Preview */}
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
                  {batch.reference_logo} ({batch.infringement_count} infringements, {batch.results.length} results ≥{currentMinThreshold}%)
                </option>
              ))}
            </select>
          </div>
          
          <div className="reference-preview">
            <h4>Reference Image Preview:</h4>
            <div className="reference-image-container">
              <img 
                src={referenceImageUrl} 
                alt={`Reference: ${currentBatch.reference_logo}`}
                className="reference-preview-image"
                onError={(e) => {
                  e.target.style.display = 'none';
                  e.target.nextSibling.style.display = 'block';
                }}
              />
              <div className="image-error-fallback" style={{display: 'none'}}>
                <p>Error loading reference image</p>
                <small>{currentBatch.reference_logo}</small>
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
            <option value="image_score">Image Score</option>
            <option value="color_similarity">Color Similarity</option>
            <option value="vit_similarity">ViT Similarity</option>
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

        {/* Dynamic threshold adjustment */}
        <div className="filter-group">
          <label>Min Similarity:</label>
          <input
            type="number"
            value={currentMinThreshold}
            onChange={(e) => setCurrentMinThreshold(parseFloat(e.target.value))}
            min="0"
            max="100"
            step="1"
            style={{ width: '80px' }}
          />
          <span>%</span>
        </div>
      </div>

      {/* Results Summary */}
      <div className="results-summary">
        <p>
          Showing <strong>{sortedResults.length}</strong> results with similarity ≥ {currentMinThreshold}% 
          for reference image: <strong>{currentBatch.reference_logo}</strong>
        </p>
        {currentBatch.results.length === 0 && (
          <div className="no-results-message">
            <p>⚠️ No results found above {currentMinThreshold}% similarity threshold.</p>
            <p>Try lowering the minimum similarity threshold to see more results.</p>
          </div>
        )}
      </div>

      <div className="comparison-grid">
        {sortedResults.map((result, index) => (
          <ResultCard 
            key={index}
            result={result}
            comparisonFolderPath={data.comparison_folder_path}
            referenceImageUrl={referenceImageUrl}
            rank={index + 1}
          />
        ))}
      </div>
    </div>
  );
};

// Updated Result Card Component with rank
const ResultCard = ({ result, comparisonFolderPath, referenceImageUrl, rank }) => {
  const [showDetails, setShowDetails] = useState(false);
  const [imageError, setImageError] = useState({ ref: false, comp: false });
  
  // Helper function to construct proper image URLs
  const constructImageUrl = (folderPath, imageName) => {
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
      {/* Header with Rank */}
      <div className="card-header-new">
        <div className="card-title-section">
          <div className="rank-badge">#{rank}</div>
          <div className="status-icon">
            {result.infringement_detected ? '🚨' : '🛡️'}
          </div>
          <span className="card-title">{result.logo_name}</span>
        </div>
        <div className={`infringement-badge ${result.infringement_detected ? 'active' : ''}`}>
          {result.infringement_detected ? 'INFRINGEMENT' : 'CLEAR'}
        </div>
      </div>

      {/* Side-by-side Image Comparison */}
      <div className="image-comparison-section">
        <div className="comparison-overlay">
          <span className="vs-text">vs {result.logo_name}</span>
        </div>
        
        <div className="images-container">
          {/* Reference Image */}
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

          {/* Divider Line */}
          <div className="divider-line"></div>

          {/* Comparison Image */}
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
        <div className={`score-badge ${result.final_similarity >= 70 ? 'high-score' : 'low-score'}`}>
          {result.final_similarity.toFixed(1)}%
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="metrics-grid">
        <div className="metric-item">
          <span className="metric-label">Text:</span>
          <span className="metric-value">{result.text_similarity.toFixed(1)}%</span>
        </div>
        <div className="metric-item">
          <span className="metric-label">Color:</span>
          <span className="metric-value">{result.color_similarity.toFixed(1)}%</span>
        </div>
        <div className="metric-item">
          <span className="metric-label">ViT:</span>
          <span className="metric-value">{result.vit_similarity.toFixed(1)}%</span>
        </div>
        <div className="metric-item">
          <span className="metric-label">Image:</span>
          <span className="metric-value">{result.image_score.toFixed(1)}%</span>
        </div>
      </div>

      {/* Extracted Text Section */}
      <div className="extracted-text-section">
        <div className="section-header">
          <span className="text-icon">📝</span>
          <span>Extracted Text</span>
        </div>
        <div className="text-comparison">
          <div className="text-row">
            <span className="text-label">Reference:</span>
            <span className="text-content">"{result.text1 || 'No text detected'}"</span>
          </div>
          <div className="text-row">
            <span className="text-label">Comparison:</span>
            <span className="text-content">"{result.text2 || 'No text detected'}"</span>
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

      {/* Additional Details (if expanded) */}
      {showDetails && (
        <div className="additional-details">
          <div className="detail-item">
            <strong>File Path:</strong>
            <p>{result.logo_path}</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
