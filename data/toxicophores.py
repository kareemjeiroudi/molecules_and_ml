toxicophores = {
    'Specific aromatic nitro': 'O=N(~O)a',
    'Specific aromatic amine': 'a[NH2]',
    'aromatic nitroso': 'a[N;X2]=O',
    'alkyl nitrite': 'CO[N;X2]=O',
    'nitrosamine': 'N[N;X2]=O',
    'epoxide': 'O1[c,C]-[c,C]1',
    'aziridine': 'C1NC1',
    'azide': 'N=[N+1]=[N-]',
    'diazo': 'C=[N+]=[N-]',
    'triazene': 'N=N-n',
    'aromatic azo': 'c[N;X2]!@;=[N;X2]c',
    'aromatic azoxy': 'cN!@;=[N;X3](O)c',
    'unsubstituted heteroatom-bonded heteroatom': '[OH,NH2][N,O]',
    'hydroperoxide': '[OH]O',
    'oxime': '[OH][N;X2]=C',
    '1,2-disubstituted peroxide': '[c,C]OO[c,C]',
    '1,2-disubstituted aliphatic hydrazine': 'C[NH][NH]C',
    'aromatic hydroxylamine': '[OH]Na',
    'aliphatic hydroxylamine': '[OH]N',
    'aromatic hydrazine': '[NH2]Na',
    'aliphatic hydrazine': '[NH2]N',
    'diazohydroxyl': '[OH][N;X2]=[N;X2]',
    'aliphatic halide': '[Cl,Br,I]C',
    'carboxylic acide halide': '[Cl,Br,I]C=O',
    'nitrogen or sulphur mustarg': '[N,S]!@[C;X4]!@[CH2][Cl,Br,I]',
    'aliphatic monohalide': '[Cl,Br,I][C;X4]',
    'α-cholorthioalkane': 'SC[Cl]',
    'β-halo ethoxy group': '[Cl,Br,I]!@[C;X4]!@[C;X4]O',
    'chloroalkene':  '[Cl]C([X1])=C[X1]',
    '1-chloroethyl': '[Cl,Br,I][CH][CH3]',
    'polyhaloalkene': '[Cl,Br,I]C(([F,Cl,Br,I])[X1])C=C',
    'polyhalocarbonyl': '[Cl,Br,I]C(([F,Cl,Br,I])[X1])C(=O)[c,C]',
    'bay-region in Polycyclic Aromatic Hydrocarbons': '[cH]1[cH]ccc2c1c3c(cc2)cc[cH][cH]3',
    'K-region in Polycyclic Aromatic Hydrocarbons': '[cH]1cccc2c1[cH][cH]c3c2ccc[cH]3',
    'polycyclic aromatic system': '' # I don't understand this one
}

additional_toxicophores = {
    'sulphonate-bonded carbon (alkyl alkane sulphonate or dialkyl sulphate)': '[$([C,c]OS((=O)=O)O!@[c,C]),$([c,C]S((=O)=O)O!@[c,C])]',
    'aliphatic N-nitro': 'O=N(~O)N',
    'α,β unsaturated aldehyde (including α-carbonyl aldehyde)': '[$(O=[CH]C=C),$(O=[CH]C=O)]',
    'diazonium': '[N;v4]#N',
    'β-propiolactone': 'O=C1CCO1',
    'α,β unsaturated alkoxy group': '[CH]=[CH]O',
    '1-aryl-2-monoalkyl hydrazine': '[NH;!R][NH;!R]a',
    'aromatic methylamine': '[CH3][NH]a',
    'ester derivative of aromatic hydroaxylamine (including original specific toxicophore)': 'aN([$([OH]),$(O*=O)])[$([#1]),$(C(=O)[CH3]),$([CH3]),$([OH]),$(O*=O)]',
    'polycyclic planar system': '' # Nor this one
}