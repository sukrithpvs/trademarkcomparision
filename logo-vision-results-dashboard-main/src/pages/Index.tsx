
import { useState, useMemo } from "react";
import { Search, Filter, Download, BarChart3, Shield, AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import ResultCard from "@/components/ResultCard";
import StatsOverview from "@/components/StatsOverview";

// Mock data structure
const mockData = {
  batch_results: [
    {
      reference_logo: "nike_logo.png",
      results: [
        {
          logo_path: "/placeholder.svg",
          logo_name: "nike_comparison.png",
          text_similarity: 95.5,
          color_similarity: 87.3,
          vit_similarity: 92.1,
          image_score: 91.2,
          final_similarity: 91.8,
          infringement_detected: true,
          text1: "NIKE",
          text2: "NIKE"
        },
        {
          logo_path: "/placeholder.svg",
          logo_name: "adidas_logo.png",
          text_similarity: 25.5,
          color_similarity: 45.3,
          vit_similarity: 30.1,
          image_score: 33.2,
          final_similarity: 33.8,
          infringement_detected: false,
          text1: "NIKE",
          text2: "ADIDAS"
        },
        {
          logo_path: "/placeholder.svg",
          logo_name: "puma_logo.png",
          text_similarity: 15.2,
          color_similarity: 52.1,
          vit_similarity: 28.4,
          image_score: 31.9,
          final_similarity: 31.9,
          infringement_detected: false,
          text1: "NIKE",
          text2: "PUMA"
        }
      ]
    },
    {
      reference_logo: "apple_logo.png", 
      results: [
        {
          logo_path: "/placeholder.svg",
          logo_name: "apple_variant.png",
          text_similarity: 88.7,
          color_similarity: 91.2,
          vit_similarity: 89.5,
          image_score: 89.8,
          final_similarity: 89.8,
          infringement_detected: true,
          text1: "Apple",
          text2: "Apple"
        },
        {
          logo_path: "/placeholder.svg",
          logo_name: "microsoft_logo.png",
          text_similarity: 12.3,
          color_similarity: 23.8,
          vit_similarity: 18.9,
          image_score: 18.3,
          final_similarity: 18.3,
          infringement_detected: false,
          text1: "Apple",
          text2: "Microsoft"
        }
      ]
    }
  ]
};

export interface LogoResult {
  logo_path: string;
  logo_name: string;
  text_similarity: number;
  color_similarity: number;
  vit_similarity: number;
  image_score: number;
  final_similarity: number;
  infringement_detected: boolean;
  text1: string;
  text2: string;
  reference_logo?: string;
}

const Index = () => {
  const [searchTerm, setSearchTerm] = useState("");
  const [sortBy, setSortBy] = useState("final_similarity_desc");
  const [filterBy, setFilterBy] = useState("all");
  const [currentPage, setCurrentPage] = useState(1);
  const resultsPerPage = 6;

  // Flatten all results for easier processing
  const allResults = useMemo(() => {
    return mockData.batch_results.flatMap(batch => 
      batch.results.map(result => ({
        ...result,
        reference_logo: batch.reference_logo
      }))
    );
  }, []);

  // Apply filters and sorting
  const filteredAndSortedResults = useMemo(() => {
    let filtered = allResults.filter(result => {
      const matchesSearch = result.logo_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           result.text1.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           result.text2.toLowerCase().includes(searchTerm.toLowerCase());
      
      const matchesFilter = filterBy === "all" || 
                           (filterBy === "infringement" && result.infringement_detected) ||
                           (filterBy === "clear" && !result.infringement_detected);
      
      return matchesSearch && matchesFilter;
    });

    // Sort results
    filtered.sort((a, b) => {
      switch (sortBy) {
        case "final_similarity_desc":
          return b.final_similarity - a.final_similarity;
        case "final_similarity_asc":
          return a.final_similarity - b.final_similarity;
        case "text_similarity_desc":
          return b.text_similarity - a.text_similarity;
        case "color_similarity_desc":
          return b.color_similarity - a.color_similarity;
        default:
          return b.final_similarity - a.final_similarity;
      }
    });

    return filtered;
  }, [allResults, searchTerm, sortBy, filterBy]);

  // Pagination
  const totalPages = Math.ceil(filteredAndSortedResults.length / resultsPerPage);
  const paginatedResults = filteredAndSortedResults.slice(
    (currentPage - 1) * resultsPerPage,
    currentPage * resultsPerPage
  );

  const handleExport = () => {
    console.log("Exporting results...");
    // Export functionality would be implemented here
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-600 rounded-lg">
                <BarChart3 className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Logo Similarity Analysis Results</h1>
                <p className="text-sm text-gray-600">Comprehensive trademark infringement detection</p>
              </div>
            </div>
            <Button onClick={handleExport} variant="outline" className="flex items-center gap-2">
              <Download className="h-4 w-4" />
              Export Results
            </Button>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Statistics Overview */}
        <StatsOverview results={allResults} />

        {/* Filters and Controls */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Filter className="h-5 w-5" />
              Filters & Search
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col lg:flex-row gap-4">
              <div className="flex-1">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                  <Input
                    placeholder="Search by logo name or text content..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10"
                  />
                </div>
              </div>
              
              <Select value={sortBy} onValueChange={setSortBy}>
                <SelectTrigger className="w-full lg:w-64">
                  <SelectValue placeholder="Sort by..." />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="final_similarity_desc">Final Score (High to Low)</SelectItem>
                  <SelectItem value="final_similarity_asc">Final Score (Low to High)</SelectItem>
                  <SelectItem value="text_similarity_desc">Text Similarity (High to Low)</SelectItem>
                  <SelectItem value="color_similarity_desc">Color Similarity (High to Low)</SelectItem>
                </SelectContent>
              </Select>

              <Select value={filterBy} onValueChange={setFilterBy}>
                <SelectTrigger className="w-full lg:w-48">
                  <SelectValue placeholder="Filter by..." />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Results</SelectItem>
                  <SelectItem value="infringement">Infringements Only</SelectItem>
                  <SelectItem value="clear">Clear Only</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardContent>
        </Card>

        {/* Results Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6 mb-8">
          {paginatedResults.map((result, index) => (
            <ResultCard
              key={`${result.reference_logo}-${result.logo_name}-${index}`}
              result={result}
            />
          ))}
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex justify-center items-center gap-2">
            <Button
              variant="outline"
              onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
              disabled={currentPage === 1}
            >
              Previous
            </Button>
            
            {Array.from({ length: totalPages }, (_, i) => i + 1).map(page => (
              <Button
                key={page}
                variant={currentPage === page ? "default" : "outline"}
                onClick={() => setCurrentPage(page)}
                className="w-10"
              >
                {page}
              </Button>
            ))}
            
            <Button
              variant="outline"
              onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
              disabled={currentPage === totalPages}
            >
              Next
            </Button>
          </div>
        )}

        {/* No Results */}
        {filteredAndSortedResults.length === 0 && (
          <Card className="text-center py-12">
            <CardContent>
              <AlertTriangle className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-gray-900 mb-2">No Results Found</h3>
              <p className="text-gray-600">Try adjusting your search terms or filters.</p>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
};

export default Index;
