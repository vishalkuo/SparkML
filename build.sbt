name := "MovieLensRecommender"

version := "1.0"

scalaVersion := "2.11.7"


libraryDependencies  ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.6.0"
)

resolvers ++= Seq(
  // other resolvers here
  // if you want to use snapshot builds (currently 0.12-SNAPSHOT), use this.
  "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)