DOCTYPE html>
    <!-- <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Formulario de Metadatos de Balística del NIST</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            /* Estilo para una apariencia más prolija en los campos numéricos */
            input[type="number"]::-webkit-inner-spin-button,
            input[type="number"]::-webkit-outer-spin-button {
                -webkit-appearance: none;
                margin: 0;
            }
            input[type="number"] {
                -moz-appearance: textfield;
            }
            .section-content {
                transition: max-height 0.5s ease-in-out;
                overflow: hidden;
            }
        </style>
    </head>
    <body class="bg-gray-900 text-gray-200 font-sans"> -->

        <div class="container mx-auto p-4 md:p-8">
            <header class="text-center mb-10">
                <h1 class="text-4xl font-bold text-white">Formulario de Metadatos de Balística</h1>
                <p class="text-lg text-gray-400 mt-2">Basado en la especificación del NIST para la Base de Datos de Marcas de Herramientas Balísticas</p>
            </header>

            <!-- El formulario ahora apunta a la ruta '/submit' con el método POST -->
            <form action="/submit" method="POST" class="space-y-8">

                <!-- Sección 1: Estudio -->
                <section class="bg-gray-800 p-6 rounded-lg shadow-lg">
                    <h2 class="text-2xl font-semibold text-white border-b border-gray-700 pb-3 mb-6 flex justify-between items-center cursor-pointer section-toggle">
                        <span>1. Estudio</span>
                        <svg class="w-6 h-6 transform transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
                    </h2>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6 section-content">
                        <div>
                            <label for="studyId" class="block text-sm font-medium text-gray-400 mb-2">StudyID (ID de Estudio)</label>
                            <input type="text" id="studyId" name="studyId" class="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500" placeholder="Generado por la base de datos" disabled>
                        </div>
                        <div>
                            <label for="studyName" class="block text-sm font-medium text-gray-400 mb-2">StudyName (Nombre del Estudio) <span class="text-red-500">*</span></label>
                            <input type="text" id="studyName" name="studyName" class="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500" required>
                        </div>
                        <div class="md:col-span-2">
                            <label for="description" class="block text-sm font-medium text-gray-400 mb-2">Description (Descripción)</label>
                            <textarea id="description" name="description" rows="3" class="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"></textarea>
                        </div>
                        <div class="md:col-span-2">
                            <label for="literatureReference" class="block text-sm font-medium text-gray-400 mb-2">LiteratureReference (Referencia Bibliográfica)</label>
                            <textarea id="literatureReference" name="literatureReference" rows="3" class="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"></textarea>
                        </div>
                        <div class="flex items-center">
                            <input type="checkbox" id="hasPersistence" name="hasPersistence" value="true" class="h-5 w-5 bg-gray-700 border-gray-600 rounded text-indigo-500 focus:ring-indigo-500">
                            <label for="hasPersistence" class="ml-3 text-sm text-gray-300">HasPersistence (Contiene disparos de persistencia)</label>
                        </div>
                        <div class="flex items-center">
                            <input type="checkbox" id="hasConsecutive" name="hasConsecutive" value="true" class="h-5 w-5 bg-gray-700 border-gray-600 rounded text-indigo-500 focus:ring-indigo-500">
                            <label for="hasConsecutive" class="ml-3 text-sm text-gray-300">HasConsecutive (Contiene componentes consecutivos)</label>
                        </div>
                        <div class="flex items-center">
                            <input type="checkbox" id="hasDifferentAmmo" name="hasDifferentAmmo" value="true" class="h-5 w-5 bg-gray-700 border-gray-600 rounded text-indigo-500 focus:ring-indigo-500">
                            <label for="hasDifferentAmmo" class="ml-3 text-sm text-gray-300">HasDifferentAmmo (Contiene diferente munición)</label>
                        </div>
                    </div>
                </section>

                <!-- Sección 2: Creador -->
                <section class="bg-gray-800 p-6 rounded-lg shadow-lg">
                    <h2 class="text-2xl font-semibold text-white border-b border-gray-700 pb-3 mb-6 flex justify-between items-center cursor-pointer section-toggle">
                        <span>2. Creador</span>
                        <svg class="w-6 h-6 transform transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
                    </h2>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6 section-content">
                        <div>
                            <label for="creatorId" class="block text-sm font-medium text-gray-400 mb-2">CreatorID (ID de Creador)</label>
                            <input type="text" id="creatorId" name="creatorId" class="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500" placeholder="Generado por la base de datos" disabled>
                        </div>
                        <div>
                            <label for="firstName" class="block text-sm font-medium text-gray-400 mb-2">FirstName (Nombre) <span class="text-red-500">*</span></label>
                            <input type="text" id="firstName" name="firstName" class="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500" required>
                        </div>
                        <div>
                            <label for="lastName" class="block text-sm font-medium text-gray-400 mb-2">LastName (Apellido) <span class="text-red-500">*</span></label>
                            <input type="text" id="lastName" name="lastName" class="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500" required>
                        </div>
                        <div>
                            <label for="organization" class="block text-sm font-medium text-gray-400 mb-2">Organization (Organización) <span class="text-red-500">*</span></label>
                            <input type="text" id="organization" name="organization" class="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500" required>
                        </div>
                    </div>
                </section>
                
                <!-- Sección 3: Arma de Fuego -->
                <section class="bg-gray-800 p-6 rounded-lg shadow-lg">
                    <h2 class="text-2xl font-semibold text-white border-b border-gray-700 pb-3 mb-6 flex justify-between items-center cursor-pointer section-toggle">
                        <span>3. Arma de Fuego</span>
                        <svg class="w-6 h-6 transform transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
                    </h2>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6 section-content">
                        <div>
                            <label for="firearmId" class="block text-sm font-medium text-gray-400 mb-2">FirearmID (ID Arma)</label>
                            <input type="text" id="firearmId" name="firearmId" class="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500" placeholder="Generado por la base de datos" disabled>
                        </div>
                        <div>
                            <label for="firearmName" class="block text-sm font-medium text-gray-400 mb-2">FirearmName (Nombre Arma) <span class="text-red-500">*</span></label>
                            <input type="text" id="firearmName" name="firearmName" class="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500" required>
                        </div>
                        <div>
                            <label for="brand" class="block text-sm font-medium text-gray-400 mb-2">Brand (Marca) <span class="text-red-500">*</span></label>
                            <select id="brand" name="brand" class="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500" required>
                                <option value="">Seleccione una marca</option>
                                <option>Beretta</option> <option>Bersa</option> <option>Browning</option> <option>Canik</option> <option>Chiappa</option> <option>Colt</option> <option>CZ-USA</option> <option>FN Herstal</option> <option>Glock</option> <option>Heckler & Koch</option> <option>Hi-Point</option> <option>Kahr Arms</option> <option>Kel-Tec</option> <option>Kimber</option> <option>Para Ordnance</option> <option>Remington</option> <option>Rossi</option> <option>Ruger</option> <option>Sig Sauer</option> <option>Smith & Wesson</option> <option>Springfield Armory</option> <option>Steyr Arms</option> <option>Stoeger</option> <option>Taurus</option> <option>Walther</option> <option>Other</option>
                            </select>
                        </div>
                        <div>
                            <label for="model" class="block text-sm font-medium text-gray-400 mb-2">Model (Modelo) <span class="text-red-500">*</span></label>
                            <input type="text" id="model" name="model" class="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500" required>
                        </div>
                        <div>
                            <label for="caliber" class="block text-sm font-medium text-gray-400 mb-2">Caliber (Calibre) <span class="text-red-500">*</span></label>
                            <select id="caliber" name="caliber" class="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500" required>
                                <option value="">Seleccione un calibre</option>
                                <option>22LR</option> <option>25 Auto</option> <option>32 Auto</option> <option>9 mm</option> <option>38/357</option> <option>357 Sig</option> <option>380 Auto</option> <option>40/10 mm</option> <option>44 Spl/Mag</option> <option>45 Auto</option> <option>Other</option>
                            </select>
                        </div>
                        <div>
                            <label for="firingPinClass" class="block text-sm font-medium text-gray-400 mb-2">FiringPinClass (Clase de Percutor)</label>
                            <select id="firingPinClass" name="firingPinClass" class="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500">
                                <option value="">Seleccione una clase</option>
                                <option>Hemispherical</option> <option>Glock Type</option> <option>Truncated Cone</option> <option>Rectangular</option> <option>Other</option> <option>Not Specified</option>
                            </select>
                        </div>
                        <div>
                            <label for="breechFaceClass" class="block text-sm font-medium text-gray-400 mb-2">BreechFaceClass (Clase de Culote)</label>
                            <select id="breechFaceClass" name="breechFaceClass" class="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500">
                            <option value="">Seleccione una clase</option>
                                <option>Arched</option> <option>Circular</option> <option>Cross Hatch</option> <option>Granular</option> <option>Smooth</option> <option>Striated</option> <option>Other</option> <option>Not Specified</option>
                            </select>
                        </div>
                        <div>
                            <label for="numberOfLands" class="block text-sm font-medium text-gray-400 mb-2">NumberOfLands (Número de Estrías)</label>
                            <select id="numberOfLands" name="numberOfLands" class="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500">
                                <option value="">Seleccione un número</option>
                                <option>2</option> <option>3</option> <option>4</option> <option>5</option> <option>6</option> <option>7</option> <option>8</option> <option>9</option> <option>≥10</option> <option>Not specified</option>
                            </select>
                        </div>
                        <div>
                            <label for="twistDirection" class="block text-sm font-medium text-gray-400 mb-2">TwistDirection (Dirección de Torsión)</label>
                            <select id="twistDirection" name="twistDirection" class="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500">
                                <option value="">Seleccione una dirección</option>
                                <option>Left</option> <option>Right</option> <option>Not specified</option>
                            </select>
                        </div>
                    </div>
                </section>

                <!-- Sección 4: Bala -->
                <section class="bg-gray-800 p-6 rounded-lg shadow-lg">
                    <h2 class="text-2xl font-semibold text-white border-b border-gray-700 pb-3 mb-6 flex justify-between items-center cursor-pointer section-toggle">
                        <span>4. Bala</span>
                        <svg class="w-6 h-6 transform transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
                    </h2>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6 section-content">
                        <div>
                            <label for="specimenName" class="block text-sm font-medium text-gray-400 mb-2">SpecimenName (Nombre Espécimen) <span class="text-red-500">*</span></label>
                            <input type="text" id="specimenName" name="specimenName" class="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500" required>
                        </div>
                        <div>
                            <label for="bulletBrand" class="block text-sm font-medium text-gray-400 mb-2">Brand (Marca Munición) <span class="text-red-500">*</span></label>
                            <select id="bulletBrand" name="bulletBrand" class="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500" required>
                                <option value="">Seleccione una marca</option>
                                <option>Aguila</option> <option>Bear</option> <option>CCI</option> <option>Federal</option> <option>Fiocchi</option> <option>FN</option> <option>Hornady</option> <option>Nosler</option> <option>PMC</option> <option>Remington</option> <option>Sellier & Bellot</option> <option>Speer</option> <option>Tulammo</option> <option>Weatherby</option> <option>Winchester</option> <option>Wolf</option> <option>Other</option>
                            </select>
                        </div>
                        <div>
                            <label for="weight" class="block text-sm font-medium text-gray-400 mb-2">Weight (Peso en granos)</label>
                            <select id="weight" name="weight" class="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500">
                                <option value="">Seleccione un rango</option>
                                <option>30-50</option> <option>51-100</option> <option>101-150</option> <option>151-200</option> <option>201-250</option> <option>251-300</option> <option>> 301</option> <option>Not Specified</option>
                            </select>
                        </div>
                        <div>
                            <label for="surfaceMaterial" class="block text-sm font-medium text-gray-400 mb-2">SurfaceMaterial (Material Superficie)</label>
                            <select id="surfaceMaterial" name="surfaceMaterial" class="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500">
                            <option value="">Seleccione un material</option>
                                <option>Copper</option> <option>Brass</option> <option>Steel</option> <option>Lead</option> <option>Polymer</option> <option>Other</option> <option>Not specified</option>
                            </select>
                        </div>
                    </div>
                </section>

                <!-- Sección 5: Medición de la Bala -->
                <section class="bg-gray-800 p-6 rounded-lg shadow-lg">
                    <h2 class="text-2xl font-semibold text-white border-b border-gray-700 pb-3 mb-6 flex justify-between items-center cursor-pointer section-toggle">
                        <span>5. Medición de la Bala</span>
                        <svg class="w-6 h-6 transform transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
                    </h2>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6 section-content">
                        <div>
                            <label for="fileName" class="block text-sm font-medium text-gray-400 mb-2">FileName (Nombre de Archivo) <span class="text-red-500">*</span></label>
                            <input type="text" id="fileName" name="fileName" class="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500" required>
                        </div>
                        <div>
                            <label for="measurementType" class="block text-sm font-medium text-gray-400 mb-2">MeasurementType (Tipo de Medición) <span class="text-red-500">*</span></label>
                            <select id="measurementType" name="measurementType" class="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg" required>
                                <option>Disc Scanning Confocal</option> <option>Laser Scanning Confocal</option> <option>Interferometry</option> <option>Focus Variation</option> <option>Photo Stereometric</option> <option>Micro Array Confocal</option> <option>Reflectance Microscopy</option> <option>Stylus</option> <option>Other</option>
                            </select>
                        </div>
                        <div>
                            <label for="measurand" class="block text-sm font-medium text-gray-400 mb-2">Measurand (Muestreo) <span class="text-red-500">*</span></label>
                            <select id="measurand" name="measurand" class="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg" required>
                                <option>Photo Image</option> <option>2D Profile</option> <option>3D Topography</option>
                            </select>
                        </div>
                        <div>
                            <label for="regionOfInterest" class="block text-sm font-medium text-gray-400 mb-2">RegionOfInterest (Región de Interés) <span class="text-red-500">*</span></label>
                            <select id="regionOfInterest" name="regionOfInterest" class="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg" required>
                            <option>Land Engraved Area</option> <option>Groove Engraved Area</option> <option>Full or partial circumference</option>
                            </select>
                        </div>
                    </div>
                </section>
                
                <!-- Sección 6: Casquillo -->
                <section class="bg-gray-800 p-6 rounded-lg shadow-lg">
                    <h2 class="text-2xl font-semibold text-white border-b border-gray-700 pb-3 mb-6 flex justify-between items-center cursor-pointer section-toggle">
                        <span>6. Casquillo</span>
                        <svg class="w-6 h-6 transform transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
                    </h2>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6 section-content">
                        <div>
                            <label for="cartridgeSpecimenName" class="block text-sm font-medium text-gray-400 mb-2">SpecimenName (Nombre Espécimen) <span class="text-red-500">*</span></label>
                            <input type="text" id="cartridgeSpecimenName" name="cartridgeSpecimenName" class="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg" required>
                        </div>
                        <div>
                            <label for="caseMaterial" class="block text-sm font-medium text-gray-400 mb-2">CaseMaterial (Material del Casquillo)</label>
                            <select id="caseMaterial" name="caseMaterial" class="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg">
                                <option>Brass</option> <option>Aluminum</option> <option>Steel</option> <option>Nickel</option> <option>Other</option> <option>Not specified</option>
                            </select>
                        </div>
                        <div>
                            <label for="primerSurfaceMaterial" class="block text-sm font-medium text-gray-400 mb-2">PrimerSurfaceMaterial (Material Sup. Fulminante)</label>
                            <select id="primerSurfaceMaterial" name="primerSurfaceMaterial" class="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg">
                            <option>Brass</option> <option>Nickel</option> <option>Copper</option> <option>Other</option> <option>Not Specified</option>
                            </select>
                        </div>
                    </div>
                </section>

                <!-- Sección 7: Medición del Casquillo -->
                <section class="bg-gray-800 p-6 rounded-lg shadow-lg">
                    <h2 class="text-2xl font-semibold text-white border-b border-gray-700 pb-3 mb-6 flex justify-between items-center cursor-pointer section-toggle">
                        <span>7. Medición del Casquillo</span>
                        <svg class="w-6 h-6 transform transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
                    </h2>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6 section-content">
                        <div class="flex items-center">
                            <input type="checkbox" id="hasBreechFace" name="hasBreechFace" value="true" class="h-5 w-5 bg-gray-700 border-gray-600 rounded text-indigo-500 focus:ring-indigo-500">
                            <label for="hasBreechFace" class="ml-3 text-sm text-gray-300">HasBreechFace (Incluye impresión de culote)</label>
                        </div>
                        <div class="flex items-center">
                            <input type="checkbox" id="hasFiringPin" name="hasFiringPin" value="true" class="h-5 w-5 bg-gray-700 border-gray-600 rounded text-indigo-500 focus:ring-indigo-500">
                            <label for="hasFiringPin" class="ml-3 text-sm text-gray-300">HasFiringPin (Incluye impresión de percutor)</label>
                        </div>
                        <div class="flex items-center">
                            <input type="checkbox" id="hasEjectorMark" name="hasEjectorMark" value="true" class="h-5 w-5 bg-gray-700 border-gray-600 rounded text-indigo-500 focus:ring-indigo-500">
                            <label for="hasEjectorMark" class="ml-3 text-sm text-gray-300">HasEjectorMark (Incluye marca de expulsor)</label>
                        </div>
                        <div class="flex items-center">
                            <input type="checkbox" id="hasApertureShear" name="hasApertureShear" value="true" class="h-5 w-5 bg-gray-700 border-gray-600 rounded text-indigo-500 focus:ring-indigo-500">
                            <label for="hasApertureShear" class="ml-3 text-sm text-gray-300">HasApertureShear (Incluye cizalladura de apertura)</label>
                        </div>
                    </div>
                </section>
                
                <div class="flex justify-end pt-4">
                    <button type="submit" class="bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-6 rounded-lg shadow-md transition duration-300 ease-in-out">
                        Enviar Datos
                    </button>
                </div>

            </form>

        </div>

        <script>
            document.querySelectorAll('.section-toggle').forEach(header => {
                const content = header.nextElementSibling;
                const icon = header.querySelector('svg');

                // Iniciar todas las secciones cerradas excepto la primera
                const sectionTitle = header.querySelector('span').textContent.trim();
                if (sectionTitle !== '1. Estudio') {
                    content.style.maxHeight = '0px';
                    icon.style.transform = 'rotate(-90deg)';
                } else {
                    content.style.maxHeight = content.scrollHeight + 'px';
                }
                
                header.addEventListener('click', () => {
                    if (content.style.maxHeight && content.style.maxHeight !== '0px') {
                        content.style.maxHeight = '0px';
                        icon.style.transform = 'rotate(-90deg)';
                    } else {
                        // Recalcular scrollHeight por si el contenido ha cambiado dinámicamente
                        content.style.maxHeight = content.scrollHeight + 'px';
                        icon.style.transform = 'rotate(0deg)';
                    }
                });
            });
        </script>

    </body>
    </html>

 -->