# Deeplearning eksamensnoter, efterår 2019

* 8 minuters eksamen

## Overall learning goals
* Identify and describe visual recognition tasks that can be solved with deep learning.
* Describe and compare different neural networks architectures.
* Explain and compare techniques for training neural networks.
* Apply deep learning to standard visual recognition tasks and interpret the results.
* Design your own course project, implement it, experiment with it, analyse and relate the results to techniques and theories in deep learning for visual recognition.

## Generalle ting at vide
_Baseret på lektion 1 og 2_.

### Forskel på AI, ML og DL

* Artificial Intelligence: Er det generelle term, om en computer der virker som en menneskelig hjerne
* Machine Learning: er en sub-kategori og kræver meget manuel arbejde. Man definere f.eks selv sine _features_, og kan så træne. Mange parametre.
* Deep Learning: er en sub-kategori af machine learning. Her kaster man bare billeder og labels efter en struktur og træner; der er (relativt) ikke så meget man kan juster på, og det hele foregår meget som en blackbox. Består af neurale netværk, der via sine layers og connections imellem dem definere et billede.

### Image classification

* _Image classification_ er en måde at klassificere hvad et billede repræsentere
* Det har mange udfordringer: 
	* Viewpoint kan være mærkelig
	* Belysning kan være dårlig
	* Skala kan være underlig
	* Noget kan være deformt
	* Noget kan være gemt væk
	* Det kan være mærkelige designs
	* Der kan være dårlig baggrund 
* At man bruger en _data driven_ tilgang, betyder at vi i kode ikke kan specificere at koden skal genkende en hund, men derimod kan vi træne på data der består af labeled billeder
* _Pipelinen_ for sådan et trick, hvor vi fra et array af pixels bestemmer en label, er som følgende:
	1. Vores input består af et sæt af _N_ billeder hver med mulighed for at være en af _M_ forskellige klasser.
	2. Vi _træner_ så vores _classifier_ på dette _trainingset_ til at kende hver klasse i sættet.
	3. Til sidst _evaluer_ vi på utrænede billede, hvor god den er til det.
* Et _trainingset_ er til at træne en model
* Et _testset_ er til evaluering af modellens kunnen når den er trænet 
* _K-Nearest Neighbours_
	* Er en simpel måde at klassificer data på
	* Man gemmer data og label for billeder
	* Når man vil _predicte_ hvilken label ny data har, ser man blot hvilket data der ligner mest og bruger den label.
		* Lad os forestille os, at alt data er plottet som et punkt hver i et koordinat system, så vil vi forestille os at de punkter der har samme _label_ ligger ved siden af hinanden.
		* Plot nu noget nyt data for at finde dens _label_
		* Hvis _K_ er _1_, så bruges label fra den der ligger tættest på, da denne ligner mest
		* Hvis _K_ er _\>1_, og de _K_ tætteste der ligger på kommer fra flere forskellige label, så er det den type som er flest af
	* Man skal prøve sig frem for at finde en god _K_
		* Er K for stor, kan små grupperinger blive ignoreret til fordel for store
	* Til at definere sådan længder, kan bruges to funktioner:
		* L1 finder den mindste absolutte forskel, og er ikke sensitiv til outliers
		* L2 finder kvadratiske fejlrate, og er sensitiv til outliers
	* KNN tager ingen tid at træne, da den bare gemmer data. Men den tager O(n) at predicate.
* _Hyperparamter_
	* Er parameter I forhold til modellen som vi sætter _før_ træning, så det er forskellen på hvilke parameter vi sætter og hvilke vi træner
	* Kunne for eksempel være vores learning rate, vores epochs, vores loss funktion
	* Vi kan finde de bedste ved at spille vores dataset op i tre forskellige:
		* Training set
		* Validation set
		* Testing set
	* Vi træner så via _training set_, sætter parameter via _validation set_ og evaluere via _testing set.
	* Derved kan vi evaluere hvordan parameterne virker på data der aldrig er set før
	* Man kan også bruge _cross validation_, hvor man kan har _training_ og _test set_, men _training_ splittes op i _n_ dele og man kører så _n_ gange hvor hver del får lov at være _validation set_, og så ser man hvilke parametre der gennemsnitligt klarer sig bedst. Det er ikke brugt meget i deep learning, da det tager lang tid med de store dataset vi har i deep learning.
* _Image Features_
	* Er dele af billeder der er interrestandte
	* Simple (dårlige) er f.eks rå-pixels (som KNN bruger) eller antal af bestemte farver farver
	* I stedet kan vi lære hvilke interesandte dele af at billede, der klassikere det
	* Dette kan gøre spå to måder:
		* Supervised (labels): bruges i neurale netværk
		* Unsupervised (ingen labels): bruges I K-means clustering
			* K-means bruges til at cluster (opdele) data, f.eks ved sociale medier data, markeds analyser etc.
			* Det parter _n_ observationer i _k_ clusters, hvor en ny observation tilhører den cluster med den nærmeste _mean_
			* Algoritme:
				* Lav _k_ tilfældige cluster
				* Loop
					* Tildel en observation til den nærmeste centroid(cluster) (L1 eller L2)
					* Opdater opdeling af centroid ved at udregne mean igen
* En måde at se hvor meget to vector ligner hinanden, er ved at tage deres _inner product_. Det hentyder til hvor meget en vektor _a_ reflekter på en vektor _b_, såfremt de begge har start i origin. Er vektorerne er normaliseret til at være længde _1_, så hvis inner produktet er _1_, er de ens og er det _-1_ er de modsætninger. Matematisk angives det: $||a||\cdot||b||\cdot cos \theta
* Et billede kan ses som en vektor, hvis vi kollapser rækkerne, f.eks ved brug af numpy
* *Curse of dimensionality*
	* I K nearest neighbours virker ved at finde områder i et fearture space hvor der grupperes
	* Men haves et stort feature space, kan det virke som om at intet minder om hinanden  
* En _weight matrix_ er en matrice af vektorer man kan bruge til at reducere et feature space
	* Haves f.eks et billede der kan være 30 forskellige kategorier, svarende til en vektor af $3072 \cdot 1$, så kan det ganges med en weight vektor af $30 \cdot 3072$, for at ende med en vektor af $30 \cdot 1$. Hver indgang svarer nu til sandsynligheden for at det er en kategori, hvilket er bedre til KNN.
		* For at lave sådan en weight vektor, kunne vi f.eks bruge CIFAR10 der er en masse billeder til 10 forskellige klasser. Tages der for hver klasse en mean af alle billederne der hører til, kan den nu bruges som features.
* Har vi et billede på vector form: $x$ og en weight vector (som beskriver features) $W$ så er $Wx = x’$, der kaldes en feature vector
* Siger vi at $w$ er en vektor af vægte der beskriver features, Så skulle vi lave en _linear classifier_ der kan fortælle om et billede er noget bestemt kunne vi finde et threshold hvor at sålænge $w \cdot x$ (inner produktet) er over et bestemt threshold, så tilhører det billedet. Det betyder at plotter vi tingene, vil der komme en lige linje der beskriver hvornår et billede er en bil eller ej.
* Men en linear classifier er ikke altid det vi skal bruge, hvordan vil de f.eks opdele en donut?
	* Dog kan en non-linear tranformation af feature spaced hjælpe her. Transferer man feature spacet fra til at bruge polar koordinater vil man nu kunne separer dem med en linje