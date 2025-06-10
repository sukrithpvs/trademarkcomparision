
import { useState } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Shield, ShieldAlert, Eye, FileText } from "lucide-react";
import { LogoResult } from "@/pages/Index";

interface ResultCardProps {
  result: LogoResult;
}

const ResultCard = ({ result }: ResultCardProps) => {
  const [showDetails, setShowDetails] = useState(false);

  const getScoreColor = (score: number) => {
    if (score >= 70) return "text-green-600 bg-green-50";
    if (score >= 30) return "text-yellow-600 bg-yellow-50";
    return "text-red-600 bg-red-50";
  };

  const getScoreBadgeVariant = (score: number) => {
    if (score >= 70) return "default";
    if (score >= 30) return "secondary";
    return "destructive";
  };

  return (
    <Card className="overflow-hidden hover:shadow-lg transition-all duration-300 group">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {result.infringement_detected ? (
              <ShieldAlert className="h-5 w-5 text-red-500" />
            ) : (
              <Shield className="h-5 w-5 text-green-500" />
            )}
            <span className="font-semibold text-sm truncate">
              {result.logo_name}
            </span>
          </div>
          <Badge 
            variant={result.infringement_detected ? "destructive" : "default"}
            className="font-semibold"
          >
            {result.infringement_detected ? "INFRINGEMENT" : "CLEAR"}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Simple Image Display */}
        <div className="relative">
          <div className="rounded-lg overflow-hidden bg-gray-100 aspect-[16/9] flex">
            <div className="w-1/2 relative border-r border-gray-300">
              <img 
                src="/placeholder.svg" 
                alt="Reference Logo"
                className="w-full h-full object-contain"
                onError={(e) => {
                  const target = e.target as HTMLImageElement;
                  target.src = "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgZmlsbD0iI2Y1ZjVmNSIvPjx0ZXh0IHg9IjUwIiB5PSI1NSIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OTk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSI+UmVmZXJlbmNlPC90ZXh0Pjwvc3ZnPg==";
                }}
              />
              <div className="absolute bottom-2 left-2 bg-black/70 text-white text-xs px-2 py-1 rounded">
                Reference
              </div>
            </div>
            <div className="w-1/2 relative">
              <img 
                src={result.logo_path} 
                alt="Comparison Logo"
                className="w-full h-full object-contain"
                onError={(e) => {
                  const target = e.target as HTMLImageElement;
                  target.src = "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgZmlsbD0iI2Y1ZjVmNSIvPjx0ZXh0IHg9IjUwIiB5PSI1NSIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OTk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSI+Q29tcGFyaXNvbjwvdGV4dD48L3N2Zz4=";
                }}
              />
              <div className="absolute bottom-2 right-2 bg-black/70 text-white text-xs px-2 py-1 rounded">
                Comparison
              </div>
            </div>
          </div>
          <div className="absolute top-2 left-2 bg-black/50 text-white text-xs px-2 py-1 rounded">
            vs {result.reference_logo}
          </div>
        </div>

        {/* Similarity Scores */}
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <span className="text-sm font-medium">Final Score</span>
            <Badge variant={getScoreBadgeVariant(result.final_similarity)} className="font-bold">
              {result.final_similarity.toFixed(1)}%
            </Badge>
          </div>
          
          {showDetails && (
            <div className="space-y-2 pt-2 border-t">
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="flex justify-between">
                  <span>Text:</span>
                  <span className={`font-semibold px-2 py-1 rounded ${getScoreColor(result.text_similarity)}`}>
                    {result.text_similarity.toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Color:</span>
                  <span className={`font-semibold px-2 py-1 rounded ${getScoreColor(result.color_similarity)}`}>
                    {result.color_similarity.toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>ViT:</span>
                  <span className={`font-semibold px-2 py-1 rounded ${getScoreColor(result.vit_similarity)}`}>
                    {result.vit_similarity.toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Image:</span>
                  <span className={`font-semibold px-2 py-1 rounded ${getScoreColor(result.image_score)}`}>
                    {result.image_score.toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* Extracted Text */}
              <div className="pt-2 border-t space-y-1">
                <div className="flex items-center gap-1 text-xs font-medium text-gray-600">
                  <FileText className="h-3 w-3" />
                  Extracted Text
                </div>
                <div className="text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-500">Reference:</span>
                    <span className="font-medium">"{result.text1}"</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Comparison:</span>
                    <span className="font-medium">"{result.text2}"</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Toggle Details Button */}
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setShowDetails(!showDetails)}
          className="w-full flex items-center gap-2 text-xs"
        >
          <Eye className="h-3 w-3" />
          {showDetails ? "Hide Details" : "Show Details"}
        </Button>
      </CardContent>
    </Card>
  );
};

export default ResultCard;
