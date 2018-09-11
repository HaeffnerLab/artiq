use board_misoc::{csr, config};
use core::fmt;

pub const DEST_COUNT: usize = 256;
pub const MAX_HOPS: usize = 32;
pub const INVALID_HOP: u8 = 0xff;

pub struct RoutingTable([[u8; MAX_HOPS]; DEST_COUNT]);

impl RoutingTable {
    // default routing table is for star topology with no hops
    fn default_master(default_n_links: usize) -> RoutingTable {
        let mut ret = RoutingTable([[INVALID_HOP; MAX_HOPS]; DEST_COUNT]);
        for i in 0..default_n_links {
            ret.0[i][0] = i as u8;
        }
        for i in 1..default_n_links {
            ret.0[i][1] = 0x00;
        }
        ret
    }

    // satellites receive the routing table from the master
    // by default, block everything
    fn default_satellite() -> RoutingTable {
        RoutingTable([[INVALID_HOP; MAX_HOPS]; DEST_COUNT])
    }
}

impl fmt::Display for RoutingTable {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RoutingTable {{")?;
        for i in 0..DEST_COUNT {
            if self.0[i][0] != INVALID_HOP {
                write!(f, "{}:", i)?;
                for j in 0..MAX_HOPS {
                    if self.0[i][j] == INVALID_HOP {
                        break;
                    }
                    write!(f, " {}", self.0[i][j])?;
                }
                write!(f, ";")?;
            }
        }
        write!(f, " }}")?;
        Ok(())
    }
}

pub fn config_routing_table(default_n_links: usize) -> RoutingTable {
    let mut ret = RoutingTable::default_master(default_n_links);
    let ok = config::read("routing_table", |result| {
        if let Ok(data) = result {
            if data.len() == DEST_COUNT*MAX_HOPS {
                for i in 0..DEST_COUNT {
                    for j in 0..MAX_HOPS {
                        ret.0[i][j] = data[i*MAX_HOPS+j];
                    }
                }
                return true;
            }
        }
        false
    });
    if !ok {
        warn!("could not read routing table from configuration, using default");
    } 
    ret
}

pub fn program_interconnect(rt: &RoutingTable, rank: u8)
{
    for i in 0..DEST_COUNT {
        let hop = rt.0[i][rank as usize];
        unsafe {
            csr::cri_con::routing_destination_write(i as _);
            csr::cri_con::routing_hop_write(hop);
        }
    }
}
