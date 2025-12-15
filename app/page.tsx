"use client"

import type React from "react"
import jsPDF from "jspdf"

import { useState } from "react"
import { Upload, Dna as DNA, Brain, FileText, AlertTriangle, CheckCircle, Clock, Download } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Alert, AlertDescription } from "@/components/ui/alert"

interface AnalysisResult {
  id: string
  variant: string
  chromosome: string
  position: number
  riskLevel: "high" | "medium" | "low"
  condition: string
  confidence: number
  description: string
  recommendations: string[]
}

export default function DNAAnalyzer() {
  const [file, setFile] = useState<File | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [results, setResults] = useState<AnalysisResult[]>([])
  const [analysisComplete, setAnalysisComplete] = useState(false)

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFile = event.target.files?.[0]
    if (uploadedFile) {
      setFile(uploadedFile)
      setResults([])
      setAnalysisComplete(false)
    }
  }

  const exportToPDF = () => {
    if (results.length === 0) return

    const doc = new jsPDF()

    // Set up PDF styling
    doc.setFontSize(20)
    doc.text("DNA Variant Analysis Report", 20, 30)

    doc.setFontSize(12)
    doc.text(`Generated: ${new Date().toLocaleDateString()}`, 20, 45)

    // Summary section
    doc.setFontSize(16)
    doc.text("SUMMARY", 20, 65)

    doc.setFontSize(11)
    const summaryText = [
      `Total variants analyzed: ${results.length}`,
      `High risk variants: ${results.filter((r) => r.riskLevel === "high").length}`,
      `Medium risk variants: ${results.filter((r) => r.riskLevel === "medium").length}`,
      `Low risk variants: ${results.filter((r) => r.riskLevel === "low").length}`,
    ]

    summaryText.forEach((text, index) => {
      doc.text(text, 20, 80 + index * 8)
    })

    // Detailed results section
    doc.setFontSize(16)
    doc.text("DETAILED RESULTS", 20, 120)

    let yPosition = 135

    results.forEach((result, index) => {
      // Check if we need a new page
      if (yPosition > 250) {
        doc.addPage()
        yPosition = 30
      }

      doc.setFontSize(14)
      doc.text(`${index + 1}. ${result.condition}`, 20, yPosition)
      yPosition += 10

      doc.setFontSize(10)
      doc.text(`Variant: ${result.variant}`, 25, yPosition)
      yPosition += 6
      doc.text(`Location: Chr${result.chromosome}:${result.position.toLocaleString()}`, 25, yPosition)
      yPosition += 6
      doc.text(`Risk Level: ${result.riskLevel.toUpperCase()}`, 25, yPosition)
      yPosition += 6
      doc.text(`Confidence: ${result.confidence}%`, 25, yPosition)
      yPosition += 10

      // Description with text wrapping
      const descriptionLines = doc.splitTextToSize(result.description, 160)
      doc.text("Description:", 25, yPosition)
      yPosition += 6
      doc.text(descriptionLines, 25, yPosition)
      yPosition += descriptionLines.length * 5 + 5

      // Recommendations
      doc.text("Recommendations:", 25, yPosition)
      yPosition += 6
      result.recommendations.forEach((rec) => {
        const recLines = doc.splitTextToSize(`• ${rec}`, 155)
        doc.text(recLines, 30, yPosition)
        yPosition += recLines.length * 5 + 2
      })

      yPosition += 10
    })

    // Add disclaimer on new page if needed
    if (yPosition > 220) {
      doc.addPage()
      yPosition = 30
    }

    doc.setFontSize(12)
    doc.text("IMPORTANT DISCLAIMER", 20, yPosition)
    yPosition += 10

    doc.setFontSize(10)
    const disclaimerText = doc.splitTextToSize(
      "These results are for research purposes only and should not be used for clinical decision-making without consultation with a qualified healthcare provider or genetic counselor.",
      160,
    )
    doc.text(disclaimerText, 20, yPosition)

    // Save the PDF
    doc.save(`DNA_Analysis_Report_${new Date().toISOString().split("T")[0]}.pdf`)
  }

  const analyzeWithML = async () => {
    if (!file) return

    setIsAnalyzing(true)
    setProgress(0)

    // Create form data
    const formData = new FormData()
    formData.append("file", file)

    // Show progress updates
    const progressInterval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 90) {
          clearInterval(progressInterval)
          return 90
        }
        return prev + 10
      })
    }, 1000)

    console.log("[v0] Sending file to ML analysis API...")

    try {
      // Call real ML analysis API
      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      })

      clearInterval(progressInterval)
      setProgress(100)

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const result = await response.json()
      console.log("[v0] ML analysis result:", result)

      if (result.error) {
        throw new Error(result.error)
      }

      if (result.pathogenic_variants && result.pathogenic_variants.length > 0) {
        const convertedResults: AnalysisResult[] = result.pathogenic_variants.map((variant: any, index: number) => ({
          id: (index + 1).toString(),
          variant: `${variant.gene} variant`,
          chromosome: variant.chromosome?.toString() || "Unknown",
          position: variant.position || 0,
          riskLevel: (variant.pathogenicity_score > 0.8
            ? "high"
            : variant.pathogenicity_score > 0.6
              ? "medium"
              : "low") as "high" | "medium" | "low",
          condition: variant.disease || "Unknown condition",
          confidence: Math.round((variant.pathogenicity_score || 0) * 100),
          description: `Pathogenic variant in ${variant.gene} gene associated with ${variant.disease}.`,
          recommendations: [
            "Consult with a genetic counselor to discuss results",
            "Consider additional family screening if indicated",
            "Regular monitoring and preventive care as recommended",
            "Discuss treatment options with healthcare providers",
          ],
        }))

        setResults(convertedResults)
      } else {
        // No variants found - this is a valid result, not an error
        setResults([])
      }

      setAnalysisComplete(true)
      setFile(null)
      // Reset file input
      const fileInput = document.getElementById("file-upload") as HTMLInputElement
      if (fileInput) fileInput.value = ""
    } catch (error) {
      console.error("[v0] Analysis failed:", error)
      alert(`Analysis failed: ${error.message}`)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const simulateAnalysis = async () => {
    if (!file) return

    setIsAnalyzing(true)
    setProgress(0)

    // Simulate ML analysis with progress updates
    const steps = [
      { progress: 20, message: "Preprocessing DNA sequences..." },
      { progress: 40, message: "Running variant calling algorithms..." },
      { progress: 60, message: "Applying machine learning models..." },
      { progress: 80, message: "Analyzing pathogenicity scores..." },
      { progress: 100, message: "Generating clinical interpretations..." },
    ]

    for (const step of steps) {
      await new Promise((resolve) => setTimeout(resolve, 1500))
      setProgress(step.progress)
    }

    // Simulate analysis results
    const mockResults: AnalysisResult[] = [
      {
        id: "1",
        variant: "BRCA1 c.5266dupC",
        chromosome: "17",
        position: 41197694,
        riskLevel: "high",
        condition: "Hereditary Breast and Ovarian Cancer",
        confidence: 94.2,
        description:
          "Pathogenic frameshift variant in BRCA1 gene associated with significantly increased risk of breast and ovarian cancer.",
        recommendations: [
          "Genetic counseling recommended",
          "Enhanced screening protocols",
          "Consider prophylactic measures",
          "Family cascade testing advised",
        ],
      },
      {
        id: "2",
        variant: "APOE ε4/ε4",
        chromosome: "19",
        position: 45411941,
        riskLevel: "medium",
        condition: "Alzheimer's Disease Risk",
        confidence: 87.6,
        description: "Homozygous APOE ε4 variant associated with increased risk of late-onset Alzheimer's disease.",
        recommendations: [
          "Lifestyle modifications for brain health",
          "Regular cognitive assessments",
          "Cardiovascular risk management",
          "Consider research participation",
        ],
      },
      {
        id: "3",
        variant: "CYP2D6*4/*4",
        chromosome: "22",
        position: 42126611,
        riskLevel: "medium",
        condition: "Drug Metabolism Variant",
        confidence: 91.8,
        description: "Poor metabolizer phenotype for CYP2D6-metabolized medications including many psychiatric drugs.",
        recommendations: [
          "Pharmacogenomic testing confirmation",
          "Medication dosing adjustments",
          "Alternative drug selection",
          "Inform healthcare providers",
        ],
      },
      {
        id: "4",
        variant: "HFE C282Y/H63D",
        chromosome: "6",
        position: 26093141,
        riskLevel: "low",
        condition: "Hereditary Hemochromatosis",
        confidence: 76.3,
        description: "Compound heterozygous variants with mild to moderate iron overload risk.",
        recommendations: [
          "Iron studies monitoring",
          "Dietary iron awareness",
          "Regular health check-ups",
          "Family screening consideration",
        ],
      },
    ]

    setResults(mockResults)
    setIsAnalyzing(false)
    setAnalysisComplete(true)
    setFile(null)
    // Reset file input
    const fileInput = document.getElementById("file-upload") as HTMLInputElement
    if (fileInput) fileInput.value = ""
  }

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case "high":
        return "bg-destructive text-destructive-foreground"
      case "medium":
        return "bg-warning text-warning-foreground"
      case "low":
        return "bg-success text-success-foreground"
      default:
        return "bg-muted text-muted-foreground"
    }
  }

  const getRiskIcon = (risk: string) => {
    switch (risk) {
      case "high":
        return <AlertTriangle className="h-4 w-4" />
      case "medium":
        return <Clock className="h-4 w-4" />
      case "low":
        return <CheckCircle className="h-4 w-4" />
      default:
        return null
    }
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <DNA className="h-8 w-8 text-primary" />
              <h1 className="text-2xl font-bold text-foreground">GenomeAI</h1>
            </div>
            <Badge variant="secondary" className="ml-2">
              Beta
            </Badge>
          </div>
          <p className="text-muted-foreground mt-2 text-balance">
            Advanced machine learning platform for detecting genetic variants and disease markers from DNA sequencing
            data
          </p>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="max-w-6xl mx-auto space-y-8">
          {/* Upload Section */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Upload className="h-5 w-5" />
                Upload DNA Sequencing Data
              </CardTitle>
              <CardDescription>
                Upload your VCF, FASTQ, or other genomic data files for analysis. Supported formats: .vcf, .fastq,
                .fasta, .bam
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="border-2 border-dashed border-border rounded-lg p-8 text-center hover:border-primary/50 transition-colors">
                  <input
                    type="file"
                    id="file-upload"
                    className="hidden"
                    accept=".vcf,.fastq,.fasta,.bam,.gz"
                    onChange={handleFileUpload}
                  />
                  <label htmlFor="file-upload" className="cursor-pointer">
                    <FileText className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                    <p className="text-lg font-medium mb-2">{file ? file.name : "Choose a file or drag and drop"}</p>
                    <p className="text-sm text-muted-foreground">Maximum file size: 50MB</p>
                  </label>
                </div>

                {file && (
                  <div className="flex items-center justify-between p-4 bg-muted rounded-lg">
                    <div className="flex items-center gap-3">
                      <FileText className="h-5 w-5 text-primary" />
                      <div>
                        <p className="font-medium">{file.name}</p>
                        <p className="text-sm text-muted-foreground">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                      </div>
                    </div>
                    <Button onClick={analyzeWithML} disabled={isAnalyzing} className="flex items-center gap-2">
                      <Brain className="h-4 w-4" />
                      {isAnalyzing ? "Analyzing..." : "Start ML Analysis"}
                    </Button>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Analysis Progress */}
          {isAnalyzing && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="h-5 w-5 animate-pulse" />
                  AI Analysis in Progress
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <Progress value={progress} className="w-full" />
                  <p className="text-sm text-muted-foreground text-center">
                    {progress < 20 && "Preprocessing DNA sequences..."}
                    {progress >= 20 && progress < 40 && "Running variant calling algorithms..."}
                    {progress >= 40 && progress < 60 && "Applying machine learning models..."}
                    {progress >= 60 && progress < 80 && "Analyzing pathogenicity scores..."}
                    {progress >= 80 && "Generating clinical interpretations..."}
                  </p>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Results Section */}
          {analysisComplete && (
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2">
                      <CheckCircle className="h-5 w-5 text-success" />
                      Analysis Complete
                    </CardTitle>
                    <CardDescription>
                      {results.length > 0
                        ? `Found ${results.length} significant genetic variants`
                        : "No pathogenic variants detected in the analyzed data"}
                    </CardDescription>
                  </div>
                  {results.length > 0 && (
                    <Button variant="outline" size="sm" onClick={exportToPDF}>
                      <Download className="h-4 w-4 mr-2" />
                      Export Report
                    </Button>
                  )}
                </div>
              </CardHeader>
              <CardContent>
                {results.length > 0 ? (
                  <Tabs defaultValue="variants" className="w-full">
                    <TabsList className="grid w-full grid-cols-3">
                      <TabsTrigger value="variants">Variants</TabsTrigger>
                      <TabsTrigger value="summary">Summary</TabsTrigger>
                      <TabsTrigger value="recommendations">Actions</TabsTrigger>
                    </TabsList>

                    <TabsContent value="variants" className="space-y-4">
                      {results.map((result) => (
                        <Card key={result.id} className="border-l-4 border-l-primary">
                          <CardHeader className="pb-3">
                            <div className="flex items-start justify-between">
                              <div>
                                <CardTitle className="text-lg">{result.condition}</CardTitle>
                                <CardDescription className="font-mono text-sm">
                                  {result.variant} • Chr{result.chromosome}:{result.position.toLocaleString()}
                                </CardDescription>
                              </div>
                              <div className="flex items-center gap-2">
                                <Badge className={getRiskColor(result.riskLevel)}>
                                  {getRiskIcon(result.riskLevel)}
                                  {result.riskLevel.toUpperCase()} RISK
                                </Badge>
                                <Badge variant="outline">{result.confidence}% confidence</Badge>
                              </div>
                            </div>
                          </CardHeader>
                          <CardContent className="space-y-4">
                            <div>
                              <h4 className="font-semibold text-sm mb-2">Variant Details</h4>
                              <p className="text-sm font-mono">{result.variant}</p>
                            </div>
                            <div>
                              <h4 className="font-semibold text-sm mb-2">Clinical Significance</h4>
                              <p className="text-sm text-muted-foreground">{result.description}</p>
                            </div>
                            <div>
                              <h4 className="font-semibold text-sm mb-2">Recommendations</h4>
                              <ul className="text-sm space-y-1">
                                {result.recommendations.map((rec, index) => (
                                  <li key={index} className="flex items-start gap-2">
                                    <span className="text-primary mt-1">•</span>
                                    {rec}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </TabsContent>

                    <TabsContent value="summary" className="space-y-4">
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <Card>
                          <CardContent className="p-6 text-center">
                            <div className="text-2xl font-bold text-destructive">
                              {results.filter((r) => r.riskLevel === "high").length}
                            </div>
                            <div className="text-sm text-muted-foreground">High Risk Variants</div>
                          </CardContent>
                        </Card>
                        <Card>
                          <CardContent className="p-6 text-center">
                            <div className="text-2xl font-bold text-warning">
                              {results.filter((r) => r.riskLevel === "medium").length}
                            </div>
                            <div className="text-sm text-muted-foreground">Medium Risk Variants</div>
                          </CardContent>
                        </Card>
                        <Card>
                          <CardContent className="p-6 text-center">
                            <div className="text-2xl font-bold text-success">
                              {results.filter((r) => r.riskLevel === "low").length}
                            </div>
                            <div className="text-sm text-muted-foreground">Low Risk Variants</div>
                          </CardContent>
                        </Card>
                      </div>
                    </TabsContent>

                    <TabsContent value="recommendations" className="space-y-4">
                      <Alert>
                        <AlertTriangle className="h-4 w-4" />
                        <AlertDescription>
                          <strong>Important:</strong> These results are for research purposes only and should not be
                          used for clinical decision-making without consultation with a qualified healthcare provider or
                          genetic counselor.
                        </AlertDescription>
                      </Alert>

                      <div className="space-y-4">
                        <Card>
                          <CardHeader>
                            <CardTitle className="text-lg">Immediate Actions</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <ul className="space-y-2 text-sm">
                              <li className="flex items-start gap-2">
                                <span className="text-primary mt-1">•</span>
                                Schedule genetic counseling consultation
                              </li>
                              <li className="flex items-start gap-2">
                                <span className="text-primary mt-1">•</span>
                                Share results with your primary healthcare provider
                              </li>
                              <li className="flex items-start gap-2">
                                <span className="text-primary mt-1">•</span>
                                Consider confirmatory testing through clinical laboratory
                              </li>
                            </ul>
                          </CardContent>
                        </Card>

                        <Card>
                          <CardHeader>
                            <CardTitle className="text-lg">Long-term Monitoring</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <ul className="space-y-2 text-sm">
                              <li className="flex items-start gap-2">
                                <span className="text-primary mt-1">•</span>
                                Establish enhanced screening protocols based on risk factors
                              </li>
                              <li className="flex items-start gap-2">
                                <span className="text-primary mt-1">•</span>
                                Discuss family cascade testing options
                              </li>
                              <li className="flex items-start gap-2">
                                <span className="text-primary mt-1">•</span>
                                Stay informed about emerging treatments and research
                              </li>
                            </ul>
                          </CardContent>
                        </Card>
                      </div>
                    </TabsContent>
                  </Tabs>
                ) : (
                  <div className="text-center py-8 space-y-4">
                    <div className="mx-auto w-16 h-16 bg-success/10 rounded-full flex items-center justify-center">
                      <CheckCircle className="h-8 w-8 text-success" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold">No Pathogenic Variants Detected</h3>
                      <p className="text-muted-foreground mt-2 max-w-md mx-auto">
                        The ML analysis completed successfully but did not identify any pathogenic variants associated
                        with the following conditions in your genetic data:
                      </p>
                      <div className="mt-4 text-sm text-muted-foreground max-w-lg mx-auto">
                        <ul className="grid grid-cols-1 md:grid-cols-2 gap-1 text-left">
                          <li>• Hereditary Breast Cancer</li>
                          <li>• Li-Fraumeni Syndrome</li>
                          <li>• Cystic Fibrosis</li>
                          <li>• Huntington's Disease</li>
                          <li>• Marfan Syndrome</li>
                          <li>• Alzheimer's Disease</li>
                          <li>• Hypertrophic Cardiomyopathy</li>
                        </ul>
                      </div>
                    </div>
                    <Alert className="max-w-md mx-auto">
                      <AlertTriangle className="h-4 w-4" />
                      <AlertDescription>
                        This result indicates that no concerning variants were found for the analyzed conditions.
                        However, this does not rule out all genetic risks. Consult with a genetic counselor for
                        comprehensive evaluation.
                      </AlertDescription>
                    </Alert>
                  </div>
                )}
              </CardContent>
            </Card>
          )}
        </div>
      </main>
    </div>
  )
}
