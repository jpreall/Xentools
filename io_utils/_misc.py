import pandas as pd


def write_xenium_gene_groups(xen_gene_list_dict, output_file):
    """
    Writes a dictionary of gene lists to a CSV file that is 
    can be read by Xenium Explorer as a gene group file.
    
    Parameters
    ----------
    xen_gene_list_dict : dict
        A dictionary where keys are group names and values are lists of gene names.
        Should have the format:
        example_dict = {
        'ADM' : ['Trim29','Aqp3','Ctse'],
        'Acinar' : ['Try10','Prss3','Gatm'],
        'Ductal' : ['Cp','Tm4sf4','Krt8'],
        'Test' : ['Cp','Try10','Aqp3'], }

    output_file : str
        The path to the output CSV file where the gene groups will be written.
    """

    inverted_dict = _invert_xen_gene_list_dict(xen_gene_list_dict)

    with open(output_file, 'w', newline='') as f:
        f.write('gene,group\n')
        for gene, groups in inverted_dict.items():
            f.write(f"{gene},{','.join(groups)}\n")
            
def _invert_xen_gene_list_dict(xen_gene_list_dict):
    """
    Inverts a dictionary of gene lists to a dictionary where each gene maps to a list of sets.
    """
    from collections import defaultdict
    gene_to_sets = defaultdict(list)
    for set_name, genes in xen_gene_list_dict.items():
        for gene in genes:
            gene_to_sets[gene].append(set_name)
    return gene_to_sets


def extract_ome_channel_names(path):
    """
    Extract OME channel names from a TIFF file's OME-XML metadata.
    Parameters
    ----------
    path : str
        Path to the TIFF file.

    Returns
    -------
    pd.Series
        A pandas Series containing the channel names.
    """
    import tifffile
    import xml.etree.ElementTree as ET
    with tifffile.TiffFile(path) as tif:
        xml = tif.ome_metadata  # raw XML string

    root = ET.fromstring(xml)
    ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}

    channels = [
        ch.attrib.get("Name")
        for ch in root.findall(".//ome:Image[1]/ome:Pixels/ome:Channel", ns)
    ]

    return pd.Series(channels, name="channel")
