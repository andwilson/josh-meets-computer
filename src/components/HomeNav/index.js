import React from 'react';
import Link from 'gatsby-link'
import styled from 'styled-components';

import boy from './boy.png';
import computer from './computer.png';

const Wrapper = styled.div`
  width: 300px;
`;

const TitleBar = styled.div`
  display: flex;
  justify-content: space-between;
  height: 50px;
  margin-bottom: 10px;
`;

const NavBar = styled.div`
  display: flex;
  justify-content: space-between;
  height: 50px;
`;

const Img = styled.img`
  max-height: 90%;
  max-width: 90%;
`;

export default () => (
  <Wrapper>
    <TitleBar>
      <Img src={boy} />
      <h2>Josh Zastrow</h2>
      <Img src={computer} />
    </TitleBar>
    <NavBar>
      <Link to={'/projects/'}>Projects</Link>
      <Link to={'/notes/'}>Notes</Link>
      <Link to={'/letters/'}>Letters</Link>
      <Link to={'/about/'}>About</Link>
    </NavBar>
  </Wrapper>
);
