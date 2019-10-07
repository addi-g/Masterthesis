main_pricing_func = function(datei,basedir){
  # Erzeugt, durch Angabe des absoluten Pfads der Input-Datei und des Grundverzeichnis 
  # welchen den Ordner "code" und "data" beinhaltet, wobei "code" die benötigten R-files enthält, 
  # eine neue Datei in "data", die die geschätzten Optionspreise enthält.
  #
  # 1.  Monte-Carlo-Schätzer werden nach dem Longstaff-Schwartz (LS) 
  #     oder Tsitsiklis-Van-Roy (TR) Algorithmus berechnet.
  # 2.  Monte-Carlo-Schätzer werden auf neu simulierten Pfade durch 
  #     die plug-in Schätzer der optimalen Stoppzeit angewendet.
  # 3.  Das arithmetische Mittel wird über die geschätzten Optionspreise gebildet
  #     und als endgültig geschätzter Optionspreis des jeweiligen Monte-Carlo-Schätzers verwendet.
  # 4.  Die Schritte 1. bis 3. werden mehrfach wiederholt und diese Optionspreise werden in einer Datei gespeichert
  
  setwd(basedir)
  source("./code/mcarlo.R")
  source("./code/in_out.R")
  source("./code/help_funcs.R")
  input   = input_func(paste(datei)) # Auslesen der Input-Datei.
  # Die Bedeutung der Werte ist aus "amerput_mce_func" zu entnehmen.
  w       = as.numeric(input["w"])
  spot    = as.numeric(input["spot"])
  strike  = as.numeric(input["strike"])
  n       = as.numeric(input["n"])
  m       = as.numeric(input["m"])
  r       = as.numeric(input["r"])
  T       = as.numeric(input["T"])
  sigma   = as.numeric(input["sigma"])
  K       = as.numeric(input["K"])
  algo    = as.character(input["algo"])
  u       = as.numeric(input["u"])
  prices  = numeric(w)
  
  for(i in (1:w)){ # Wiederholt Schritte 1. bis 3. 
    a = amerput_mce_func(spot, strike,n,m,r,T,sigma,K,algo)
    prices[i] = amerput_price_func(spot, strike, m, r, T, sigma, K, a, u)
    cat("\014")
    print(paste((i/w)*100,"% berechnet.",sep = ""))
  }
  output_func(w,spot,strike,n,m,r,T,sigma,K,algo,u,prices,paste(basedir,"/data",sep ="")) # Schreibe Ergebnisse in Output-Datei
}
