# Recogito Plugin Wrapper

An (experimental) wrapper to make the NER script available as a plugin to 
the [Recogito annotation platform](http://github.com/pelagios/recogito2).  


## Testing

To test parsing throught the Scala wrapper run

```
sbt test
```

## Compile and Deploy

- build the plugin using `sbt package`. This will create the 
  .jar file `target\recogito-plugin-ner-herodotus-0.1.jar`. 
- copy the .jar into a subfolder __inside__ the Recogito 
  `/plugins` folder
- copy the Python script and the `latin_ner` folder into the 
  same subfolder
- make sure the necessary Python dependencies are installed on 
  the Recogito server 