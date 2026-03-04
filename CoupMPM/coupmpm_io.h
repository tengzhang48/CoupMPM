#ifndef COUPMPM_IO_H
#define COUPMPM_IO_H

#include "coupmpm_grid.h"
#include "lmptype.h"
#include <mpi.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace LAMMPS_NS {
namespace CoupMPM {

// Checkpoint file format
namespace Checkpoint {
  constexpr uint32_t MAGIC   = 0x4D504D4B;  // "MPMK"
  constexpr int32_t  VERSION = 1;
}

class MPMIO {
public:

  // ==================================================================
  // Grid VTK output: .vti ImageData (cell-centered)
  // Adapted from CoupLB IO::write_vtk
  // ==================================================================
  static void write_grid_vtk(const MPMGrid& grid, MPI_Comm comm,
                             long step, const std::string& prefix,
                             const double domain_lo[3])
  {
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    const int Nx = grid.Nx, Ny = grid.Ny;
    const int Nz = (grid.dim == 3) ? grid.Nz : 1;

    const int nlx = grid.nx, nly = grid.ny;
    const int nlz = (grid.dim == 3) ? grid.nz : 1;

    const int ox = grid.offset[0];
    const int oy = grid.offset[1];
    const int oz = (grid.dim == 3) ? grid.offset[2] : 0;

    // Each rank writes its own local piece
    char fname[512];
    snprintf(fname, sizeof(fname), "%s_grid_%06ld_r%d.vti", prefix.c_str(), step, rank);
    FILE* fp = fopen(fname, "w");
    if (!fp) return;

    fprintf(fp, "<?xml version=\"1.0\"?>\n");
    fprintf(fp, "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\">\n");
    fprintf(fp, "  <ImageData WholeExtent=\"0 %d 0 %d 0 %d\" "
                "Origin=\"%.8e %.8e %.8e\" "
                "Spacing=\"%.8e %.8e %.8e\">\n",
            Nx, Ny, Nz, domain_lo[0], domain_lo[1], domain_lo[2],
            grid.dx, grid.dy, grid.dz);
    fprintf(fp, "    <Piece Extent=\"%d %d %d %d %d %d\">\n",
            ox, ox + nlx, oy, oy + nly, oz, oz + nlz);
    fprintf(fp, "      <CellData>\n");

    fprintf(fp, "        <DataArray type=\"Float64\" Name=\"mass\" format=\"ascii\">\n");
    for (int k = 0; k < nlz; k++)
      for (int j = 0; j < nly; j++)
        for (int i = 0; i < nlx; i++)
          fprintf(fp, "%.8e\n", grid.mass[grid.lidx(i, j, k)]);
    fprintf(fp, "        </DataArray>\n");

    fprintf(fp, "        <DataArray type=\"Float64\" Name=\"velocity\" "
                "NumberOfComponents=\"3\" format=\"ascii\">\n");
    for (int k = 0; k < nlz; k++)
      for (int j = 0; j < nly; j++)
        for (int i = 0; i < nlx; i++) {
          const int n = grid.lidx(i, j, k);
          fprintf(fp, "%.8e %.8e %.8e\n",
                  grid.velocity_new_x[n], grid.velocity_new_y[n], grid.velocity_new_z[n]);
        }
    fprintf(fp, "        </DataArray>\n");

    fprintf(fp, "        <DataArray type=\"Float64\" Name=\"density\" format=\"ascii\">\n");
    for (int k = 0; k < nlz; k++)
      for (int j = 0; j < nly; j++)
        for (int i = 0; i < nlx; i++)
          fprintf(fp, "%.8e\n", grid.density[grid.lidx(i, j, k)]);
    fprintf(fp, "        </DataArray>\n");

    fprintf(fp, "      </CellData>\n");
    fprintf(fp, "    </Piece>\n");
    fprintf(fp, "  </ImageData>\n");
    fprintf(fp, "</VTKFile>\n");
    fclose(fp);

    // Gather piece extents to rank 0 for master .pvti file
    int local_info[6] = {ox, oy, oz, nlx, nly, nlz};
    std::vector<int> all_info(nprocs * 6);
    MPI_Gather(local_info, 6, MPI_INT, all_info.data(), 6, MPI_INT, 0, comm);

    if (rank == 0) {
      char pvti_name[512];
      snprintf(pvti_name, sizeof(pvti_name), "%s_grid_%06ld.pvti", prefix.c_str(), step);
      FILE* pf = fopen(pvti_name, "w");
      if (!pf) return;

      fprintf(pf, "<?xml version=\"1.0\"?>\n");
      fprintf(pf, "<VTKFile type=\"PImageData\" version=\"1.0\" byte_order=\"LittleEndian\">\n");
      fprintf(pf, "  <PImageData WholeExtent=\"0 %d 0 %d 0 %d\" GhostLevel=\"0\" "
                  "Origin=\"%.8e %.8e %.8e\" "
                  "Spacing=\"%.8e %.8e %.8e\">\n",
              Nx, Ny, Nz, domain_lo[0], domain_lo[1], domain_lo[2],
              grid.dx, grid.dy, grid.dz);
      fprintf(pf, "    <PCellData>\n");
      fprintf(pf, "      <PDataArray type=\"Float64\" Name=\"mass\"/>\n");
      fprintf(pf, "      <PDataArray type=\"Float64\" Name=\"velocity\" NumberOfComponents=\"3\"/>\n");
      fprintf(pf, "      <PDataArray type=\"Float64\" Name=\"density\"/>\n");
      fprintf(pf, "    </PCellData>\n");
      for (int r = 0; r < nprocs; r++) {
        const int rox = all_info[r*6+0], roy = all_info[r*6+1], roz = all_info[r*6+2];
        const int rnx = all_info[r*6+3], rny = all_info[r*6+4], rnz = all_info[r*6+5];
        char piece_file[512];
        snprintf(piece_file, sizeof(piece_file), "%s_grid_%06ld_r%d.vti", prefix.c_str(), step, r);
        const char* base = strrchr(piece_file, '/');
        base = base ? base + 1 : piece_file;
        fprintf(pf, "    <Piece Extent=\"%d %d %d %d %d %d\" Source=\"%s\"/>\n",
                rox, rox + rnx, roy, roy + rny, roz, roz + rnz, base);
      }
      fprintf(pf, "  </PImageData>\n");
      fprintf(pf, "</VTKFile>\n");
      fclose(pf);
    }
  }

  // ==================================================================
  // Particle VTK output: .vtp PolyData (unstructured points)
  // Each rank writes its own file; a .pvtp references them all.
  // ==================================================================
  static void write_particle_vtk(int nlocal, int dim,
                                 double** x, double** v,
                                 double* stress_v, double* F_def,
                                 tagint* molecule, int* surface_flag,
                                 MPI_Comm comm,
                                 long step, const std::string& prefix)
  {
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    char fname[512];
    snprintf(fname, sizeof(fname), "%s_part_%06ld_r%d.vtp", prefix.c_str(), step, rank);
    FILE* fp = fopen(fname, "w");
    if (!fp) return;

    fprintf(fp, "<?xml version=\"1.0\"?>\n");
    fprintf(fp, "<VTKFile type=\"PolyData\" version=\"1.0\" byte_order=\"LittleEndian\">\n");
    fprintf(fp, "  <PolyData>\n");
    fprintf(fp, "    <Piece NumberOfPoints=\"%d\" NumberOfVerts=\"%d\">\n", nlocal, nlocal);

    // Points
    fprintf(fp, "      <Points>\n");
    fprintf(fp, "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n");
    for (int i = 0; i < nlocal; i++)
      fprintf(fp, "%.8e %.8e %.8e\n", x[i][0], x[i][1], (dim==3)?x[i][2]:0.0);
    fprintf(fp, "        </DataArray>\n");
    fprintf(fp, "      </Points>\n");

    // Connectivity (each point is its own vertex)
    fprintf(fp, "      <Verts>\n");
    fprintf(fp, "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n");
    for (int i = 0; i < nlocal; i++) fprintf(fp, "%d\n", i);
    fprintf(fp, "        </DataArray>\n");
    fprintf(fp, "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n");
    for (int i = 0; i < nlocal; i++) fprintf(fp, "%d\n", i+1);
    fprintf(fp, "        </DataArray>\n");
    fprintf(fp, "      </Verts>\n");

    // Point data
    fprintf(fp, "      <PointData>\n");

    // Velocity
    fprintf(fp, "        <DataArray type=\"Float64\" Name=\"velocity\" "
                "NumberOfComponents=\"3\" format=\"ascii\">\n");
    for (int i = 0; i < nlocal; i++)
      fprintf(fp, "%.8e %.8e %.8e\n", v[i][0], v[i][1], (dim==3)?v[i][2]:0.0);
    fprintf(fp, "        </DataArray>\n");

    // J = det(F)
    if (F_def) {
      fprintf(fp, "        <DataArray type=\"Float64\" Name=\"J\" format=\"ascii\">\n");
      for (int i = 0; i < nlocal; i++) {
        const double* F = &F_def[i*9];
        double J = F[0]*(F[4]*F[8]-F[5]*F[7]) - F[1]*(F[3]*F[8]-F[5]*F[6])
                 + F[2]*(F[3]*F[7]-F[4]*F[6]);
        fprintf(fp, "%.8e\n", J);
      }
      fprintf(fp, "        </DataArray>\n");
    }

    // Stress (von Mises)
    if (stress_v) {
      fprintf(fp, "        <DataArray type=\"Float64\" Name=\"vonMises\" format=\"ascii\">\n");
      for (int i = 0; i < nlocal; i++) {
        const double* s = &stress_v[i*6];
        double vm = std::sqrt(0.5*((s[0]-s[1])*(s[0]-s[1]) + (s[1]-s[2])*(s[1]-s[2])
                   + (s[2]-s[0])*(s[2]-s[0]) + 6.0*(s[3]*s[3]+s[4]*s[4]+s[5]*s[5])));
        fprintf(fp, "%.8e\n", vm);
      }
      fprintf(fp, "        </DataArray>\n");

      // Pressure
      fprintf(fp, "        <DataArray type=\"Float64\" Name=\"pressure\" format=\"ascii\">\n");
      for (int i = 0; i < nlocal; i++) {
        const double* s = &stress_v[i*6];
        fprintf(fp, "%.8e\n", -(s[0]+s[1]+s[2])/3.0);
      }
      fprintf(fp, "        </DataArray>\n");
    }

    // Body ID
    if (molecule) {
      fprintf(fp, "        <DataArray type=\"Int32\" Name=\"body_id\" format=\"ascii\">\n");
      for (int i = 0; i < nlocal; i++) fprintf(fp, "%ld\n", (long)molecule[i]);
      fprintf(fp, "        </DataArray>\n");
    }

    // Surface flag
    if (surface_flag) {
      fprintf(fp, "        <DataArray type=\"Int32\" Name=\"surface\" format=\"ascii\">\n");
      for (int i = 0; i < nlocal; i++) fprintf(fp, "%d\n", surface_flag[i]);
      fprintf(fp, "        </DataArray>\n");
    }

    fprintf(fp, "      </PointData>\n");
    fprintf(fp, "    </Piece>\n");
    fprintf(fp, "  </PolyData>\n");
    fprintf(fp, "</VTKFile>\n");
    fclose(fp);

    // Rank 0 writes .pvtp referencing all rank files
    if (rank == 0) {
      char pvtp_name[512];
      snprintf(pvtp_name, sizeof(pvtp_name), "%s_part_%06ld.pvtp", prefix.c_str(), step);
      FILE* pf = fopen(pvtp_name, "w");
      if (!pf) return;
      fprintf(pf, "<?xml version=\"1.0\"?>\n");
      fprintf(pf, "<VTKFile type=\"PPolyData\" version=\"1.0\" byte_order=\"LittleEndian\">\n");
      fprintf(pf, "  <PPolyData GhostLevel=\"0\">\n");
      fprintf(pf, "    <PPoints>\n");
      fprintf(pf, "      <PDataArray type=\"Float64\" NumberOfComponents=\"3\"/>\n");
      fprintf(pf, "    </PPoints>\n");
      fprintf(pf, "    <PPointData>\n");
      fprintf(pf, "      <PDataArray type=\"Float64\" Name=\"velocity\" NumberOfComponents=\"3\"/>\n");
      fprintf(pf, "      <PDataArray type=\"Float64\" Name=\"J\"/>\n");
      fprintf(pf, "      <PDataArray type=\"Float64\" Name=\"vonMises\"/>\n");
      fprintf(pf, "      <PDataArray type=\"Float64\" Name=\"pressure\"/>\n");
      fprintf(pf, "      <PDataArray type=\"Int32\" Name=\"body_id\"/>\n");
      fprintf(pf, "      <PDataArray type=\"Int32\" Name=\"surface\"/>\n");
      fprintf(pf, "    </PPointData>\n");
      for (int r = 0; r < nprocs; r++) {
        char piece[512];
        snprintf(piece, sizeof(piece), "%s_part_%06ld_r%d.vtp", prefix.c_str(), step, r);
        // Extract basename
        const char* base = strrchr(piece, '/');
        base = base ? base + 1 : piece;
        fprintf(pf, "    <Piece Source=\"%s\"/>\n", base);
      }
      fprintf(pf, "  </PPolyData>\n");
      fprintf(pf, "</VTKFile>\n");
      fclose(pf);
    }
  }

  // ==================================================================
  // PVD time series
  // ==================================================================
  static void write_pvd(const std::string& pvd_filename,
                        const std::string& vtk_prefix,
                        const std::vector<long>& steps,
                        double dt, const std::string& suffix = "_grid_")
  {
    FILE* fp = fopen(pvd_filename.c_str(), "w");
    if (!fp) return;

    fprintf(fp, "<?xml version=\"1.0\"?>\n");
    fprintf(fp, "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
    fprintf(fp, "  <Collection>\n");
    for (size_t i = 0; i < steps.size(); i++) {
      char vti_name[512];
      snprintf(vti_name, sizeof(vti_name), "%s%s%06ld.pvti",
               vtk_prefix.c_str(), suffix.c_str(), steps[i]);
      const char* base = strrchr(vti_name, '/');
      base = base ? base + 1 : vti_name;
      fprintf(fp, "    <DataSet timestep=\"%.8e\" file=\"%s\"/>\n",
              steps[i] * dt, base);
    }
    fprintf(fp, "  </Collection>\n");
    fprintf(fp, "</VTKFile>\n");
    fclose(fp);
  }
};

} // namespace CoupMPM
} // namespace LAMMPS_NS

#endif // COUPMPM_IO_H
