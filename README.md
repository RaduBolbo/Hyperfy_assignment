# Hyperfy_assignment

    Partea 1. Cum am gandit problema

  Exista un master care are rolul de a incarca, pe rand, frame-uri de la camera publica si care porneste doua alte doua thread-uri/procese, cate unul pentru fiecare retea de detectie. Cele doua threaduri/procese primesc cel mai recent frame si aplica inferenta cu modelul incarcat, pe acel frame, modificand fiecare cate doua variabile comune: a_boxes - o lista cu coordonatele bound box-urilor decise si a_labels - lo lista cu etichetele identificate. In tot acest timp, masterul deseneaza peste doua variante ale cadruluieului curent cit, ultimele detection box-uri produse de modele si etichetele aferente (initial nu deseneaza nimic, apan al prima inferenta reusita), apoi afiseaza cadrele respective, in ferestre diferite.
  Ca urmare, cele doua ferestre vor avea stream-ul video sincronizat si in timp real, dar detection box-urile vor avea un mic lag (pentru ca update-ul lor se face dupa viteza de inferenta a retelelor).
  Ingloband totul intr-o clasa, ar exista ca date membre instantele claselor retelelor (pe care le-am si programat) si URL-ul camerei publice. La constructor, se va prelua URL-ul si se vor instantia arhitecturile retelelor. Clasa ar fi prevazuta cu o metoda de incarcare a ponderilor si cu o alta metoda de "run", care sa declanseze toata actiunea.

    Partea 2. Ce am reusit sa fac

  Am reusit sa construiesc 3 clase pentru 3 retele diferite (3 pentru ca primele 2 aveau viteza de inferenta aproximativ egala, apoi am gasit una cu viteza diferita). Prin aceste clase se instantiaza reteaua, se incarca ponderile si se poate aplica inferenta pe o imagine. Urmeaza sa fie folosite, in cadrul programului. Am testat pe imagini functionarea lor
  Am creat un main_pillot_threads.py care sa exemplifice functionarea programului pe doua variabile simple: a si b: masterul afiseaza variabilele, iar cele doua threaduri le incrementreaza, fiecare avand un timp de sleep diferit, pentru a simula workload-ul asociat retelelor. 
  Am creat programul principal main_threads.py (dar momentan nu sub forma de clasa) care citeste frame-uri de la un videoclip si le trimite la inferenta threadurilor create. Bineinteles, am folosit lock-uri pentru a gestiona accesul la resursele comune.
  Am constatat ca thread-urile nu pot rula in paralel cu adevarat, pe core-uri distincte, in Python, asa ca am incercat acelasi lucru cu procese, acolo unde m-am lovit de probleme din cauza nepartajarii spatiului de adrese intre procese.

    Partea 3. Probleme intampinate
    
    problema 1: nu reusesc sa incarc stream-ul video. Initial primeam niste erori pe care le-am rezolvat dezactivandu-mi firewall-ul, iar acum primesc "nan", in loc de frame (chiar si dupa o perioada mai lunga, in care ar fi trebuit sa se stabileasca conexiunea). Am incercat mai multe camere online si nu a mers pentru niciuna. Am presupus ca este ceva in neregula cu sistemul de pe care rulez, asa ca am incercat si de pe un alt laptop fara succes.
    problema 2: Dupa ce am gandit problema cu threaduri, din executie, mi-am dat seama ca nu rulau in paralel. Am aflat ca GIl blocheaza executia in paralel a threadurilor, in Python, asa ca am trecut la executia cu procese. Problema curenta etse ca procesele nu partajeaza acelasi spatiu de adresa si trebuie lucrat cu niste obiecte partajate speciale, in pachetul "multiprocessing", dar nu am reusit sa fac ca modificarile asupra frame-ului, din master, sa fie vizibile si in procesele child.
    problema 3: effDet-ul preluat este teoretic antrenat pe COCO 2017, dar am incarcat etichetele acestui set de date sub forma de lista, dar se pare ca modelul produce etichete cu indecsi ce depasesc lista, asa ca, pentru moment, afisez doar index-ul obiectului detectat.
    problema 4: reteaua effDet identifica prea multe obiecte in imagine, desi am respectat formatul de input precizat pe site-ul TF de unde am luat reteaua (tipul variabilelor de intrare si range-ul). Din moment ce nu se specifica ordinea planelor de culoare, am testat pentru RGB (PIL) si BGR (cv2), dar nu am obtinut rezultate mai bune.

    Partea 4. Cum as face problema, daca as lua-o de la capat

Probabil as folosi QtDesigner, pentru ca include un pachet - PyQt5, care ofera support pentru multithreading.

In plus, ar trebui sa invat exact cum functioneaza threadurile si procesele in Python. Pana acum nu am mai lucrat cu ele decat in C#.














