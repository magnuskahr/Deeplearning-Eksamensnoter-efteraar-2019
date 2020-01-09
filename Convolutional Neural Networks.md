# Convolutional Neural Networks

{{TOC}}

## Arkitektur

Convolutional Neural Networks eller CNN for short er en kategori af neurale netværk der har vist sig at være effektive til billede genkendelse og klassifisering; her i blandt til ansigter, objekter, skilte og så videre.

For et cnn består arkitekturen af to dele: 

* første del handler om feature extraction
* anden del om klassifisering

## Feature extraction

Tager man et billede med et digital kamera, indeholder det typisk tre channels: red, green og blue.

![](rgb.png)

Ideen er nu, at bruge disse channels på en anden måde. I stedet for at de repræsentere farve af et billede, skal de repræsentere features.

Features er interresandte dele ved et billede, det kunne være en ansigts struktur, det kunne være bestemte former eller mønstre og sådan.

Vi ønsker at kunne kigge på mange forskellige features, hvorfor vi ønsker at udvide vores antal af channels i vores billede.

Ideen ved feature extraction er, at vi vil starte med at finde noget helt basale features fra et billede, og ud fra de features vil vi så finde nye features osv. osv. osv.; altså at vi abstrakter featuresne mere og mere.

* Det kunne starte med at opdage kanter i billede
* Der så kunne bruges til simple former
* Der så længere nede kunne bruges til ansigter, eller biler osv

### Convulational Layer
Til at lave feature extraction, bruger man et _convulational layer_. Det virker ved at have et filter, f.eks af størrelsen 3x3. Filteret bliver _slidet_ over vores billede, og et nyt billede bliver lavet som output på dette.

![](convlayer.png)

Billedet er jo blot en matrice af tal, og ligeså er vores filter også. Der bliver taget dot produktet af filteret, og de pixels som filteret står over, og dette bliver så en pixel i det nye billede. Så slider filteret til højre, med hvad der kaldes en _stride_, altså hvor mange pixels den skal gå til højre, for nu at udregne en ny pixel i det nye billede.

Det vil dog også sige, at størrelsen af outputtet er lidt mindre, eftersom at kanterne aldrig er i midten af filteret. Man kan udregne output størrelsen ved:

> $outputsize = \frac{inputsize - filtersize}{stride} + 1$

### Activation Function

Efter en convulation vil man ofte gerne introducere en activation funktioner. Det vil sige, at alle pixels nu skal _igennem en test_. 

> ReLU: max(0, x)

Ved f.eks at bruge ReLU, så vil alle værdier der er under 0 blive erstattet af 0. Vi kunne også bruge tanh (mapper til [-1; 1]) eller sigmoid (mapper til [0; 1])

### Pooling layer

Vi vil gerne gå endnu mere abstrakt, og et feature map indeholder meget unødvendig information. Derfor bruger vi et pooling layer, f.eks et max-pool. Et sådan layer har også et filter size og en stride. For max pooling, vil filteret tage den højeste værdi af et område og mappe det over til en ny channel. Derved får vi bare udtrukket de vigtige dele af et feature map, og har det nu bare på en mindre størrelse. 

Efterhånden som man har flere omgange af disse conv+pool layers, så søger vi at at få flere channels i vores billede, men at billedet bliver mindre og mindre. Det betyder at vi kan genkende hvad der er i billederne, men vi har desværre mistet informationen om hvor det er.

## Classification

For at bruge disse feature maps I de mange forskellige channels til at genkende forskellige ting, så bruges der nu nogle fully connected layers.

![](fullyconnected.png)

Et fully connected layer betyder, at alle units i det tager input fra alle units før det og giver output til alle efter det. I dette tilfælde betyder det, at alle units i det første fully connected layer vil modtage et input fra hver pixel i alle feature map. Hver af forbindelse har så en vægt, og når de er ganget på og alt er plusset sammen, vil en unit så bruge sin activation function til at bestemme dens output.

Det sidste lag har så lige så mange units som antallet af de kategorier man forsøger af forudsige, og ved at bruge Softmax, vil de repræsentere sandsynligheder for at inputtet svarede til den label som uniten repræsentere. Det betyder også, at output laget summer op til 1.

## Training

En ting er at bygge det her netværk, men det vil med alt sansynlighed ikke give nogle brugbare resultater før det er at vi har trænet det.

Træning handler om at justere vores vægte og filtre.

### Vægte og filtre

Først og fremmest skal vi have en start værdi for vores filtre og vores vægte. Det dur ikke at sætte dem til 0, kan det ske at der ingen asymmetri er mellem de forskellige units, hvis de har samme vægt.

Men hvad gør man så?

Hvis man bruger Sigmoid eller Tahn, kan man forsøge sig med den metode der hedder Xavier, og hvis man bruger RElU kan man prøve at bruge Kamming.

### SGD

Smides et billede fra vores træning set igennem vores netværk, vil det med alt sandsynlighed ikke forudsige den rigtige kategori. Dette er fordi at vores netværk ikke er trænet endnu, og vi vil derfor have hvad der betegnes som en høj _loss_ eller et godt dansk ord vil nok være en høj fejlrate. Denne ønske vi selvfølgelig går mod nul, hvor at netværket da vil kunne forudsige rigtigt.

Stochastic Gradient Descent, forkortet SGD, kan hjælpe os med at minimere dette loss.

Den normale _gradient descent_ udregner et gradient ud fra resultater på hele datasetet, hvor den stokastiske udgave bruger en random mini-batch af n billeder og udregner derpå. 

Men hvad er det nu lige gradient descent er?

Hvis vi siger at _loss’en_ for vores funktion er plottet og den ligner en skål, vil en tilfældig position på skålens overflade da være _loss’en_ for de vægte og filtre der er brugt der.

Vi ønsker at finde de vægte og filtre der er på bunden af skålen, hvor lossen er mindst, kaldet funktionens minimum.

For at komme derned af, laver _gradient descent_ få udregninger når langt fra den optimale løsning hvor vi har en større loss, og efterhånden som vi får en mindre loss, vil der laves flere udregninger.

Starter man et tilfældigt sted _x_ og ser hvad kurvens hældning er der, så udregnes der et _stepsize_ ud fra denne, så næste hældning måles ved $x’$. Sådan et stepsize kan man udregne ved $s = slope \cdot l$, hvor $l$ er en learning rate, f.eks. $0.1$, hvilket gør at vi søger mindre skridt.

### Problemer med SGD
Er der nogle problemer med at bruge SGD?
JA! Det kan være svært at finde en rigtig learning rate, og lossen er ikke som en simpel skål, og SGD kan f.eks have svært ved at navigere hvis den er i en kløft.

Når SGD befinder sig i en kløft, bruger den meget tid på at svinge op og ned af siderne, og kommer ikke særlig langt frem.

![](SGD.png)

Det kan man fikse ved at bruge momentum på sin SGD. Momentumet forhøjes da for dimensioner hvor gradienten peger i samme retning (ned af) og reduceres hvor gradienten skifter retning.







