
import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { BarChart3, Shield, ShieldAlert, TrendingUp } from "lucide-react";
import { LogoResult } from "@/pages/Index";

interface StatsOverviewProps {
  results: LogoResult[];
}

const StatsOverview = ({ results }: StatsOverviewProps) => {
  const stats = useMemo(() => {
    const totalComparisons = results.length;
    const infringements = results.filter(r => r.infringement_detected).length;
    const infringementRate = totalComparisons > 0 ? (infringements / totalComparisons) * 100 : 0;
    const averageScore = totalComparisons > 0 
      ? results.reduce((sum, r) => sum + r.final_similarity, 0) / totalComparisons 
      : 0;
    const highRiskCount = results.filter(r => r.final_similarity >= 70).length;

    return {
      totalComparisons,
      infringements,
      infringementRate,
      averageScore,
      highRiskCount,
      clearResults: totalComparisons - infringements
    };
  }, [results]);

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
      <Card className="border-l-4 border-l-blue-500">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Total Comparisons</CardTitle>
          <BarChart3 className="h-4 w-4 text-blue-500" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{stats.totalComparisons}</div>
          <p className="text-xs text-gray-600">Logos analyzed</p>
        </CardContent>
      </Card>

      <Card className="border-l-4 border-l-red-500">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Infringements</CardTitle>
          <ShieldAlert className="h-4 w-4 text-red-500" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold text-red-600">{stats.infringements}</div>
          <div className="flex items-center gap-2">
            <Badge variant="destructive" className="text-xs">
              {stats.infringementRate.toFixed(1)}%
            </Badge>
            <span className="text-xs text-gray-600">of total</span>
          </div>
        </CardContent>
      </Card>

      <Card className="border-l-4 border-l-green-500">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Clear Results</CardTitle>
          <Shield className="h-4 w-4 text-green-500" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold text-green-600">{stats.clearResults}</div>
          <div className="flex items-center gap-2">
            <Badge variant="default" className="text-xs bg-green-100 text-green-800">
              {((stats.clearResults / stats.totalComparisons) * 100).toFixed(1)}%
            </Badge>
            <span className="text-xs text-gray-600">safe</span>
          </div>
        </CardContent>
      </Card>

      <Card className="border-l-4 border-l-orange-500">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Avg. Similarity</CardTitle>
          <TrendingUp className="h-4 w-4 text-orange-500" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{stats.averageScore.toFixed(1)}%</div>
          <p className="text-xs text-gray-600">
            {stats.highRiskCount} high risk (≥70%)
          </p>
        </CardContent>
      </Card>
    </div>
  );
};

export default StatsOverview;
