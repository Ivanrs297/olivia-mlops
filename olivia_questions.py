WHOSE = "la persona que está siendo entrevistada"

###
### 1. identificar el último episodio de violencia
### 2. ¿Fue reciente? Sí o No
### 3. Si fue reciente: ¿hace cuánto? (abierto)
### 4. Si no fue reciente: ¿hace cuánto? (categoría)
### 5. Si no entra en categoría => otro.
### 6. ¿Cuál otro?

questions = [
    # nombre
    {
        "key": "II_nombre",
        "question": f"¿Cuál es el nombre completo de {WHOSE}?",
    },
    # genero
    {
        # TODO: identificar que tipo de género
        # set II_genero to "Otro"
        "key": "II_genero_especificar",
        "question": f"¿Con qué genero se identifica {WHOSE}? Responde 'femenino' o 'masculino'",
    },
    # sexo
    {
        "key": "II_sexo",
        "question": f"¿Cuál es el sexo de {WHOSE}? Responde 'hombre' o 'mujer'",
        "otherKey": None,
        "categories": ["mujer", "hombre", "intersexual"],
    },
    # fecha de nacimiento
    {
        "key": "II_fecha_de_nacimiento",
        "question": f"¿Qué fecha de nacimiento tiene {WHOSE}?",
    },
    # nacionalidad
    {
        "key": "II_nacionalidad",
        "question": f"¿Qué nacionalidad tiene {WHOSE}?",
    },
    # lugar de nacimiento
    {
        "key": "II_lugar_de_nacimiento",
        "question": f"¿En dónde nació {WHOSE}?",
    },
    # entidad donde reside
    {
        "key": "II_entidad_federativa_donde_reside_actualmente",
        "question": f"¿En dónde radica actualmente {WHOSE}?",
    },
    # telefono
    {
        # TODO: identificar qué tipo de teléfono
        "key": "II_telefono_fijo_casa",
        "question": f"¿Cuál es el número de contacto? Responde con el número telefónico.",
    },
    # domicilio
    {
        # TODO: dividir dirección en calle, CP, estado, municipio, colonia
        "key": "II_direccion_calle",
        "question": f"¿Cuál es el domicilio de {WHOSE}?",
    },
    # escolaridad
    {
        "key": "III_escolaridad",
        "question": f"¿Qué escolaridad tiene {WHOSE}?",
        "otherKey": None,
        "categories": [
            "kinder_o_preescolar",
            "primaria",
            "secundaria",
            "preparatoria_o_bachillerato",
            "normal",
            "carrera_tecnica_o_comercial",
            "licenciatura_o_superior",
            "posgrado",
            "ninguno",
        ],
    },
    # estatus de escolaridad
    {
        "key": "III_su_escolaridad_esta_en",
        "question": f"¿La escolaridad de {WHOSE} está terminada?",
        "otherKey": None,
        "categories": ["en_curso", "terminada", "trunca"],
    },
    # seguridad social
    {
        "key": "III_cual_seguridad_social",
        "question": f"¿Tiene seguridad social {WHOSE}?",
        "otherKey": None,
        "categories": [
            "imss",
            "issste",
            "pemex",
            "sedena",
            "insabi",
            "privado",
        ],
    },
    # ocupacion
    {
        "key": "III_ocupacion_de_la_persona",
        "question": f"¿A qué se dedica actualmente la persona entrevistada?",
        "otherKey": "III_especificar_ocupacion_de_la_persona",
        "categories": [
            "jornalera_o_albaniil",
            "empleada_o_obrera_o",
            "labores_del_hogar",
            "estudios",
            "negocio_propio",
            "deporte",
            "jubilado_pensionado",
            "ninguna",
        ],
    },
    # estado civil
    {
        "key": "III_situacion_conyugal",
        "question": f"¿Qué estado civil tiene {WHOSE}?",
        "otherKey": None,
        "categories": [
            "union_libre",
            "casada_o",
            "separada_o",
            "divorciada_o",
            "viuda_o",
            "soltera_o",
        ],
    },
    # regimen matrimonial
    {
        "key": "III_regimen_matrimonial",
        "question": f"¿Con qué régimen matrimonial está casada {WHOSE}?",
        "otherKey": None,
        "categories": [
            "separacion_de_bienes",
            "sociedad_legal",
            "sociedad_conyugal_o_voluntaria",
        ],
    },
    # tipo de vivienda
    {
        "key": "III_tipo_de_vivienda",
        "question": f"¿Cuáles son las características de la casa en la que vive {WHOSE}?",
        "otherKey": "III_especificar_tipo_de_vivienda",
        "categories": [
            "casa_independiente",
            "departamento_en_edificio_o_unidad_habitacional",
            "departamento_en_vecindad",
            "cuarto_en_la_azotea",
            "local_no_construido_para_habitacion",
            "casa_o_departamento_en_terreno_familiar",
            "casa_movil_refugio",
            "asilo",
            "orfanato_o_convento",
            "no_tiene_vivienda",
        ],
    },
    # compartida
    {
        "key": "III_compartida_con_otras_personas",
        "question": f"¿La vivienda en la que {WHOSE} es compartida?",
        "otherKey": "III_compartida_especificar",
        "categories": [
            "amistades",
            "familiares",
        ],
    },
    # cuantas personas
    {
        "key": "III_cuantas_personas_habitan_en_su_vivienda",
        "question": f"¿Cuántas personas viven en la casa de {WHOSE}?",
    },
    # quien aporta el mayor ingreso
    {
        "key": "III_quien_aporta_el_mayor_ingreso_dentro_del_hogar",
        "question": f"¿Quién aporta el mayor ingreso dentro del hogar?",
    },
    # motivo de la atencion
    {
        "key": "IV_contexto_causa_y_evolucion",
        "question": f"¿Cuál es el motivo de la atención (sé especifico)?",
    },
    # institucion medica
    {
        # TODO: set to boolean
        "key": "IV_ha_tenido_que_ser_atendida_en_una_institucion_medica_o_por_personal_medico_como_consecuencia_de_un_evento_de_violencia_con_la_persona_agresora",
        "question": f"¿Ha tenido que ser atendida en una institución médica o por personal médico como consecuencia de un evento de violencia con la persona agresora? Responde sí o no.",
        "type": "boolean",
    },
    # ultimo episodio de violencia
    {
        # TODO: set IV_ultimo_episodio_de_violencia_reciente to Boolean
        # TODO: set IV_ultimo_episodio_de_violencia_reciente_especificar if IV_ultimo_episodio_de_violencia_reciente == True
        # TODO: set IV_ultimo_episodio_de_violencia if IV_ultimo_episodio_de_violencia_reciente == False
        "key": "IV_ultimo_episodio_de_violencia_especificar",
        "question": f"¿Cuál fue el último episodio de violencia? Describelo con detalle",
    },
    # nombre PA
    {
        "key": "VI_nombre",
        "question": f"Nombre de la persona agresora",
    },
    # edad PA
    {
        "key": "VI_edad",
        "question": f"Edad de la persona agresora",
    },
    # domicilio PA
    {
        # TODO: dividir en calle, cp, colonia, etc.
        "key": "VI_calle",
        "question": f"¿Cuál es el domicilio de la persona agresora? Responde calle y colonia.",
    },
    # escolaridad PA
    {
        "key": "VI_escolaridad",
        "question": f"¿Cuál es la escolaridad de la persona agresora? Responde con el grado de estudios",
    },
    # ocupacion PA
    {
        # TODO: set VI_ocupacion_de_la_persona to categories
        "key": "VI_especificar_ocupacion",
        "question": f"¿Cuál es la ocupación de la persona agresora?",
    },
    # telefono PA
    {
        # TODO: identificar tipo de telefono
        "key": "VI_telefono_fijo_casa",
        "question": f"¿Cuál es el teléfono de la persona agresora?",
    },
    # posesion de armas PA
    {
        "key": "VI_posesion_de_armas",
        "question": f"¿La persona agresora posee armas?",
        "type": "boolean",
    },
    # crimen organizado PA
    {
        "key": "VI_pertenece_o_tiene_enlace_con_el_crimen_organizado",
        "question": f"¿La persona agresora tiene vinculos con el crimen organizado?",
        "type": "boolean",
    },
    # antecedentes penales PA
    {
        "key": "VI_historial_de_antecedentes_penales",
        "question": f"¿Qué antecedentes penales tiene la persona agresora?",
    },
    # señas particulares PA
    {
        "key": "VI_especificar_senias",
        "question": f"¿Cuáles son las características físicas que tiene la persona agresora?",
    },
]
