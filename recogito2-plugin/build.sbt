name := "recogito-plugin-ner-herodotus"

organization := "org.pelagios"

version := "0.2"

scalaVersion := "2.11.11"

scalaVersion in ThisBuild := "2.11.11"

scalacOptions += "-feature"

// Do not append Scala versions to the generated artifacts
crossPaths := false

/** Runtime dependencies **/
libraryDependencies ++= Seq(
  "org.pelagios" % "recogito-plugin-sdk" % "1.0" from "https://github.com/pelagios/recogito2-plugin-sdk/releases/download/v1.0/recogito-plugin-sdk-0.3.jar"
)

/** Test dependencies **/
libraryDependencies ++= Seq(
  "org.specs2" %% "specs2-core" % "4.3.4" % "test"
)